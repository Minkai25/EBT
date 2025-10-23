import torch
import torch.nn as nn
import lightning as L
import einops
from model.model_utils import *
from torch.nn.modules.normalization import RMSNorm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class EBTBlock(L.LightningModule):
    """
    A EBT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, latent_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Use PyTorch's built-in MultiheadAttention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads, 
            bias=True,  # equivalent to qkv_bias=True
            batch_first=True,  # expects (batch, seq, feature) format
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Simplified MLP using nn.Sequential
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 6 * hidden_size, bias=True)
        )
 
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        inside_attn = modulate(self.norm1(x), shift_msa, scale_msa)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            # nn.MultiheadAttention returns (output, attention_weights)
            attn_results, _ = self.attn(inside_attn, inside_attn, inside_attn, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_results
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of EBT.
    """
    def __init__(self, hidden_size, latent_dim, sequence_length):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.temporal_linear = nn.Linear(2 * sequence_length, 1, bias=False) # input and output sequences concatenated
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # Mean over hidden dimension 
        x = x.mean(dim=-1, keepdim=False) # (B, T)
        x = self.temporal_linear(x) # (B, 1)
        return x.squeeze(-1)


class EBT_A(nn.Module):
    """
    Bi-directional Energy Based Transformer Model for ARC tasks.
    """
    def __init__(self, 
                 height,
                 width,
                 channels,
                 hidden_dim,
                 index_embed_dim,
                 num_indices,
                 num_attn_blocks,
                 num_heads,
                 mlp_ratio):
        super().__init__()
        self.height = height
        self.width = width
        self.max_sequence_length = self.height * self.width # Currently applying same embedding to input and output grids
        self.channels = channels
        self.index_embed_dim = index_embed_dim
        self.num_indices = num_indices
        self.hidden_dim = hidden_dim
        self.num_attn_blocks = num_attn_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.index_embedding = nn.Embedding(self.num_indices, self.index_embed_dim)
        self.input_embedding = nn.Embedding(self.channels, self.hidden_dim)
        self.vocab_to_embed = nn.Linear(self.channels, self.hidden_dim, bias = False)
        self.input_offset = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.output_offset = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_sequence_length, self.hidden_dim), requires_grad=False)
        self.attn_blocks = nn.ModuleList([
            EBTBlock(hidden_size=self.hidden_dim, latent_dim=self.index_embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)
            for _ in range(self.num_attn_blocks)
        ])
        self.final_layer = FinalLayer(hidden_size=self.hidden_dim, latent_dim=self.index_embed_dim, sequence_length=self.max_sequence_length)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.max_sequence_length**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # Zero out adaLN modulation layers:
        for block in self.attn_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.temporal_linear.weight, 0)

    def forward(self, input, output, index):
        """
        Forward pass of the EBT model.
        input: (B, C, H, W) tensor representing the input grid
        output: (B, C, H, W) tensor representing the output grid
        index: (B, ) tensor representing the index of the task
        """
        input_class = torch.argmax(input, dim=1) # (B, H, W)
        input_embedding = self.input_embedding(input_class) # (B, H, W, D)
        input_sequence = einops.rearrange(input_embedding, 'b h w c -> b (h w) c') # (B, H*W, D)
        input_sequence = input_sequence + self.input_offset
        input_sequence = input_sequence + self.pos_embed[:, :input_sequence.size(1), :]
        # Use linear layer on output
        output = einops.rearrange(output, 'b c h w -> b h w c')
        output_embedding = self.vocab_to_embed(output.float()) # (B, H*W, D)
        output_sequence = einops.rearrange(output_embedding, 'b h w c -> b (h w) c') # (B, H*W, D)
        output_sequence = output_sequence + self.output_offset 
        output_sequence = output_sequence + self.pos_embed[:, :output_sequence.size(1), :]
        # print("Output sequence shape:", output_sequence.shape)

        # Combine input and output sequences
        x = torch.cat([input_sequence, output_sequence], dim=1) # (B, 2*H*W, D)
        # Embed index
        index_embed = self.index_embedding(index) # (B, D2)
        for block in self.attn_blocks:
            x = block(x, index_embed) # (B, 2*H*W, D)
        energy = self.final_layer(x, index_embed) # (B, )
        return energy


class EBT_ARC(L.LightningModule):
    """
    Energy-Based Transformer (EBT) model for ARC tasks.
    """
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        # print("height:", self.hparams.grid_height, "width:", self.hparams.grid_width, "channels:", self.hparams.grid_channels, "hidden_dim:", self.hparams.grid_hidden_dim, "index_embed_dim:", self.hparams.grid_index_embed_dim, "num_attn_blocks:", self.hparams.num_transformer_blocks, "num_heads:", self.hparams.multiheaded_attention_heads, "mlp_ratio:", self.hparams.arc_mlp_ratio)
        self.model = EBT_A(
            height=self.hparams.grid_height,
            width=self.hparams.grid_width,
            channels=self.hparams.grid_channels,
            hidden_dim=self.hparams.grid_hidden_dim,
            index_embed_dim=self.hparams.grid_index_embed_dim,
            num_indices=self.hparams.grid_num_indices,
            num_attn_blocks=self.hparams.num_transformer_blocks,
            num_heads=self.hparams.multiheaded_attention_heads,
            mlp_ratio=self.hparams.arc_mlp_ratio
        )

        self.RMSnorm = RMSNorm(normalized_shape=[self.hparams.grid_channels, self.hparams.grid_height, self.hparams.grid_width], eps=1e-6, elementwise_affine=False)
        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

        self.softmax = nn.Softmax(dim=-1)
        self.ignore_index_loss = 0 if self.hparams.ignore_padding_tokens_in_loss else -100

        # Define ARC color scheme
        # 0, 1: padding tokens (light gray, lighter gray)
        # 2: black
        # 3-11: ARC standard colors
        self.arc_colormap = self._create_arc_colormap()

    def _create_arc_colormap(self):
        """Create a colormap for ARC grids"""
        colors = [
            '#E0E0E0',  # 0: padding (grey)
            '#FFFFFF',  # 1: padding (white)
            '#000000',  # 2: black
            '#0074D9',  # 3: blue
            '#FF4136',  # 4: red
            '#2ECC40',  # 5: green
            '#FFDC00',  # 6: yellow
            '#9B59B6',  # 7: purple
            '#F012BE',  # 8: magenta
            '#FF851B',  # 9: orange
            '#7FDBFF',  # 10: cyan/teal
            '#870C25',  # 11: brown/maroon
        ]
        return mcolors.ListedColormap(colors)

    # def visualize_grid(self, inp, output, initial_pred, final_pred, idx):
    #     """
    #     Visualize input, ground truth, initial prediction, and final prediction

    #     Args:
    #         inp: Input grid (H, W) - class indices
    #         output: Ground truth output grid (H, W) - class indices
    #         initial_pred: Initial prediction grid (H, W) - class indices
    #         final_pred: Final prediction grid (H, W) - class indices
    #         idx: Sample index for identification

    #     Returns:
    #         matplotlib figure
    #     """
    #     fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    #     grids = [inp, output, initial_pred, final_pred]
    #     titles = ['Input', 'Ground Truth', 'Initial Prediction', 'Final Prediction']

    #     for ax, grid, title in zip(axes, grids, titles):
    #         im = ax.imshow(grid, cmap=self.arc_colormap, vmin=0, vmax=11, interpolation='nearest')
    #         ax.set_title(title, fontsize=12, fontweight='bold')
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)

    #     plt.suptitle(f'ARC Grid Visualization - Sample {idx}', fontsize=14, fontweight='bold')
    #     plt.tight_layout()

    #     return fig
    def forward(self, inp, output, index, learning=True): 
        predicted_distributions = []
        predicted_energies = []

        predicted_tokens = torch.randn(output.shape, device=self.device) 
        predicted_distributions.append(predicted_tokens)
        alpha = self.alpha
        # alpha = torch.clamp(self.alpha, min=0.0001)
        # alpha = torch.clamp(self.alpha, min=0.0001)
        # if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
        #     expanded_alpha = alpha.expand(batch_size, seq_length, 1)

        #     scale = self.hparams.randomize_mcmc_step_size_scale
        #     low = alpha / scale
        #     high = alpha * scale
        #     alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        # langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        langevin_dynamics_noise_std = self.langevin_dynamics_noise_std
        
        # mcmc_steps = [] # in the general case of no randomize_mcmc_num_steps then this has len == self.hparams.randomize_mcmc_num_steps
        # for step in range(self.hparams.mcmc_num_steps):
        #     if not no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0:
        #         if self.hparams.randomize_mcmc_num_steps_final_landscape: # makes so only applies rand steps to final landscape
        #             if step == (self.hparams.mcmc_num_steps - 1):
        #                 min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
        #                 repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
        #                 mcmc_steps.extend([step] * repeats)
        #             else:
        #                 mcmc_steps.append(step)
        #         else:
        #             min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
        #             repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
        #             mcmc_steps.extend([step] * repeats)
        #     elif no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0: # use max steps
        #         if step == (self.hparams.mcmc_num_steps - 1): # i found this was a better pretraining metric and was more stable, only do several steps on final energy landscape instead of over all energy landscapes
        #             mcmc_steps.extend([step] * (self.hparams.randomize_mcmc_num_steps + 1))
        #         else:
        #             mcmc_steps.append(step)
        #     else:
        #         mcmc_steps.append(step)

        with torch.set_grad_enabled(True):
            for i in range(self.hparams.mcmc_num_steps): 
                if self.hparams.no_mcmc_detach:
                    predicted_tokens.requires_grad_()
                else: # default, do detach
                    predicted_tokens = predicted_tokens.detach().requires_grad_()
                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std # langevin dynamics
                    predicted_tokens = predicted_tokens + ld_noise

                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if i == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)
                        
                #     if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                #         predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
                #     else:
                #         predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                # else:
                #     predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                energy_preds = self.model(inp, predicted_tokens, index) # output shape is B
                predicted_energies.append(energy_preds)
                
                if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                    if i == (self.hparams.mcmc_num_steps - 1):
                        predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                    else:
                        predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=False)[0]
                else:
                    predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha) 
                    # predicted_tokens_grad = scale_clamp(predicted_tokens_grad, -min_and_max, min_and_max)
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)
                    
                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")
                
                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after  
                if self.hparams.use_rmsnorm and i != (self.hparams.mcmc_num_steps - 1): # dont do rmsnorm on final step
                    predicted_tokens = self.RMSnorm(predicted_tokens)
                # if self.hparams.absolute_clamp != 0.0:
                #     predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
                
                # if self.hparams.sharpen_predicted_distribution != 0.0:
                #     predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

                # if return_raw_logits:
                #     predicted_tokens_for_loss = predicted_tokens # BS, S, V
                # else:
                #     predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
                predicted_distributions.append(predicted_tokens)        

        return predicted_distributions, predicted_energies

    def forward_loss_wrapper(self, batch, phase="train"):
        inp, output, index = batch
        no_randomness = False if phase == "train" else True
        # if not no_randomness and self.mcmc_replay_buffer: # dont do this when doing val/testing
        #     all_tokens = x['input_ids'].squeeze(dim=1)
        #     input_ids, replay_buffer_logits, next_token_indices = self.replay_buffer.get_batch(all_tokens) # this automatically does indexing for input ids and next token indices while also passing back the logits
        #     predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, replay_buffer_logits = replay_buffer_logits, no_randomness = no_randomness)
        #     self.replay_buffer.update(all_tokens.detach(), predicted_distributions[-1].detach()) # update using the final predicted distributions
        # else:
        #     input_ids = x['input_ids'].squeeze(dim=1)[:, :-1]
        #     predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness)
        #     next_token_indices = x['input_ids'].squeeze(dim=1)[:, 1:] # squeeze was to remove 1 on 2nd dim

        # if self.hparams.execution_mode == "finetune": # Only tokens after "[[Answer]]: " will be calculated in finetune
        #     next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        # next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D
        predicted_distributions, predicted_energies = self(inp, output, index, learning=(phase=="train"))
        reconstruction_loss = 0
        initial_prediction = predicted_distributions[0] # B, C, H, W
        final_prediction = predicted_distributions[-1] # B, C, H, W
        output_classes = output.argmax(dim=1) # B, H, W
        initial_loss = F.cross_entropy(initial_prediction, output_classes, reduction='mean', ignore_index=self.ignore_index_loss)
        final_reconstruction_loss = F.cross_entropy(final_prediction, output_classes, reduction='mean', ignore_index=self.ignore_index_loss)
        # # Accuracy metrics
        # with torch.no_grad():
        #     # Predicted classes from logits
        #     initial_pred_classes = initial_prediction.argmax(dim=1)  # B, H, W
        #     final_pred_classes = final_prediction.argmax(dim=1)      # B, H, W

        #     # Per-element accuracy (averaged over all elements in batch)
        #     initial_acc_per_element = (initial_pred_classes == output_classes).float().mean()
        #     final_acc_per_element = (final_pred_classes == output_classes).float().mean()

        #     # Per-grid exact-match accuracy (fraction of grids with all elements correct)
        #     B = output_classes.shape[0]
        #     initial_acc_per_grid_exact = (initial_pred_classes == output_classes).view(B, -1).all(dim=1).float().mean()
        #     final_acc_per_grid_exact = (final_pred_classes == output_classes).view(B, -1).all(dim=1).float().mean()
        
        #pure logging things (no function for training)
        initial_pred_energies = predicted_energies[0].mean().detach()
        final_pred_energies = predicted_energies[-1].mean().detach()
        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
        # total_mcmc_steps = len(predicted_energies) # in general this equals self.hparams.mcmc_num_steps, isnt in case of rand number
        # for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(predicted_distributions, predicted_energies)):
        #     if self.hparams.soften_target_prob_dist != 0.0:
        #         if total_mcmc_steps <= 1:
        #             label_smoothing = 0.0
        #         else:
        #             label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
        #         predicted_distribution = predicted_distribution.reshape(-1, self.vocab_size)
        #         cce_loss = F.cross_entropy(predicted_distribution, next_token_indices, ignore_index=self.tokenizer_pad_token_id, label_smoothing=label_smoothing)
        #     else:
        #         predicted_distribution = self.log_softmax(predicted_distribution).reshape(-1, self.vocab_size)
        #         cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=self.tokenizer_pad_token_id)
            
        #     if self.hparams.truncate_mcmc:
        #         if mcmc_step == (total_mcmc_steps - 1):
        #             reconstruction_loss = cce_loss
        #             ppl_loss = torch.exp(cce_loss).detach()
        #             final_reconstruction_loss = cce_loss.detach()
        #     else:
        #         reconstruction_loss += cce_loss
        #         if mcmc_step == (total_mcmc_steps - 1):
        #             ppl_loss = torch.exp(cce_loss).detach()
        #             final_reconstruction_loss = cce_loss.detach()
        #             reconstruction_loss = reconstruction_loss / total_mcmc_steps # normalize so is indifferent to number of mcmc steps

        log_dict = {
            'loss': final_reconstruction_loss,
            'initial_loss' : initial_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            # Accuracy logs
            # 'initial_acc_per_element': initial_acc_per_element,
            # 'final_acc_per_element': final_acc_per_element,
            # 'initial_acc_per_grid_exact': initial_acc_per_grid_exact,
            # 'final_acc_per_grid_exact': final_acc_per_grid_exact,
        }
        # Log image to wandb
        is_update_step = (self.trainer.fit_loop.epoch_loop.batch_progress.current.completed + 1) % self.trainer.accumulate_grad_batches == 0
        if self.trainer.global_step % self.hparams.log_image_every_n_steps == 0 and is_update_step and phase == "train":
            # Randomly select one sample from the batch to visualize
            batch_size = inp.shape[0]
            random_idx = torch.randint(0, batch_size, (1,)).item()

            # Convert one-hot encoded tensors to class indices and create individual visualizations
            grids_to_viz = {
                'input_image': inp[random_idx].argmax(dim=0).cpu().numpy(),
                'ground_truth_image': output[random_idx].argmax(dim=0).cpu().numpy(),
                'initial_pred_image': initial_prediction[random_idx].argmax(dim=0).cpu().numpy(),
                'final_pred_image': final_prediction[random_idx].argmax(dim=0).cpu().numpy()
            }

            # Create individual visualizations for each grid
            for key, grid in grids_to_viz.items():
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(grid, cmap=self.arc_colormap, vmin=0, vmax=11, interpolation='nearest')
                ax.set_title(key.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
                plt.tight_layout()

                # Convert to tensor for logging
                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # Convert RGBA to RGB by dropping alpha channel
                img_array = img_array[:, :, :3]
                log_dict[key] = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

                plt.close(fig)

        # Visualization during validation
        if phase == "valid":
            # Log accuracies
            with torch.no_grad():
                # Predicted classes from logits
                initial_pred_classes = initial_prediction.argmax(dim=1)  # B, H, W
                final_pred_classes = final_prediction.argmax(dim=1)      # B, H, W

                # Per-element accuracy (averaged over all elements in batch)
                initial_acc_per_element = (initial_pred_classes == output_classes).float().mean()
                final_acc_per_element = (final_pred_classes == output_classes).float().mean()

                # Per-grid exact-match accuracy (fraction of grids with all elements correct)
                B = output_classes.shape[0]
                initial_acc_per_grid_exact = (initial_pred_classes == output_classes).view(B, -1).all(dim=1).float().mean()
                final_acc_per_grid_exact = (final_pred_classes == output_classes).view(B, -1).all(dim=1).float().mean()
            log_dict.update({
                'initial_acc_per_element': initial_acc_per_element,
                'final_acc_per_element': final_acc_per_element,
                'initial_acc_per_grid_exact': initial_acc_per_grid_exact,
                'final_acc_per_grid_exact': final_acc_per_grid_exact
            })
        return log_dict