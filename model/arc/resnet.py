import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import wandb

from transformers import AutoTokenizer
from model.model_utils import *
from model.replay_buffer import CausalReplayBuffer

class ResBlock(nn.Module):
    def __init__(self, downsample=False, rescale=False, filters=64, y_dim=None, film_hidden_dim=128):
        super(ResBlock, self).__init__()
        self.filters = filters
        self.downsample = downsample
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)
        
        # FILM layers
        self.use_film = y_dim is not None
        if self.use_film:
            # MLP to generate scale and shift parameters for first conv
            self.film_mlp1 = nn.Sequential(
                nn.Linear(y_dim, film_hidden_dim),
                nn.SiLU(),
                nn.Linear(film_hidden_dim, filters * 2)
            )
            # MLP to generate scale and shift parameters for second conv
            self.film_mlp2 = nn.Sequential(
                nn.Linear(y_dim, film_hidden_dim),
                nn.SiLU(),
                nn.Linear(film_hidden_dim, filters * 2)
            )
        
        self.silu = nn.SiLU()
        
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(self, x, y=None):
        x_orig = x
        
        # First conv + FILM
        x = self.conv1(x)
        if self.use_film and y is not None:
            film_params = self.film_mlp1(y)
            scale, shift = torch.chunk(film_params, 2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            x = x * (scale + 1) + shift
        x = self.silu(x)
        
        # Second conv + FILM
        x = self.conv2(x)
        if self.use_film and y is not None:
            film_params = self.film_mlp2(y)
            scale, shift = torch.chunk(film_params, 2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            x = x * (scale + 1) + shift
        x = self.silu(x)
        
        x_out = x_orig + x
        
        if self.downsample:
            x_out = self.silu(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)
        
        return x_out
    

class GridEBM(nn.Module):
    def __init__(self, height=5, width=5, channels=10, hidden_dim=384, num_res_layers=3,
                 num_indices=1000, index_embed_dim=64, mlp_hidden_dim=128):
        super().__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.hidden_dim = hidden_dim
        # self.reduced_channels = reduced_channels

        # Index embedding
        self.index_embedding = nn.Embedding(num_indices, index_embed_dim)

        # CNN layers (same as before)
        self.conv1 = nn.Conv2d(2 * self.channels, hidden_dim, 3, padding=1)
        self.res_layers = nn.ModuleList([
            ResBlock(downsample=False, rescale=False, filters=hidden_dim, y_dim=index_embed_dim, film_hidden_dim=mlp_hidden_dim)
            for _ in range(num_res_layers)
        ])

        self.reduce_conv = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, inp, out, index):
        """
        Forward pass with index lookup

        Args:
            inp: Input grid tensor of shape (batch_size, channels, height, width)
            out: Output grid tensor of shape (batch_size, channels, height, width)
            index: Tensor of shape (batch_size,) containing indices

        Returns:
            energy: Tensor of shape (batch_size, 1) containing energy values
        """
        latents = self.index_embedding(index)  # (batch, index_embed_dim)
        # Process grids through CNN
        x = inp  # (batch, channels, height, width)
        y = out  # (batch, channels, height, width)
        x = torch.cat((x, y), dim=1)

        h = nn.SiLU()(self.conv1(x))  # (batch, hidden_dim, height, width)
        for res_layer in self.res_layers:
            h = res_layer(h, latents)  # (batch, hidden_dim, height, width)

        # Reduce channels
        output = self.reduce_conv(h)  # (batch, channels, height, width)

        # Compute energy
        energy = output.pow(2).sum(dim=[1, 2, 3], keepdim=False)  # (batch,)

        return energy

class GridEBM_ARC(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.model = GridEBM(
            height=self.hparams.grid_height,
            width=self.hparams.grid_width,
            channels=self.hparams.grid_channels,
            hidden_dim=self.hparams.grid_hidden_dim,
            num_res_layers=self.hparams.grid_num_res_layers,
            num_indices=self.hparams.grid_num_indices,
            index_embed_dim=self.hparams.grid_index_embed_dim,
            mlp_hidden_dim=self.hparams.grid_mlp_hidden_dim
        )

        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

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

                # if self.hparams.normalize_initial_condition:
                #     if self.hparams.normalize_initial_condition_only_first_step:
                #         if mcmc_step == 0:
                #             predicted_tokens = self.softmax(predicted_tokens)
                #     else:
                #         predicted_tokens = self.softmax(predicted_tokens)
                        
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
        initial_loss = F.cross_entropy(initial_prediction, output_classes, reduction='mean')
        final_reconstruction_loss = F.cross_entropy(final_prediction, output_classes, reduction='mean')
        
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

        # Visualization during validation
        if phase == "valid":
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