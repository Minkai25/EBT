from typing import Tuple, List, Dict, Any, Sequence
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from model.arc.trm.common import trunc_normal_init_
from model.arc.trm.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from model.arc.trm.sparse_embedding import CastedSparseEmbedding
from model.arc.trm.losses import stablemax_cross_entropy

IGNORE_LABEL_ID = -100
PAD_ID = 0
BLANK_IDENTIFIER_ID = 0

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))
        # breakpoint()
        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=self.H_init.device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=self.L_init.device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        # # Debug tensor devices
        # import pdb; pdb.set_trace()
        # print(f"reset_flag device: {reset_flag.device}")
        # print(f"self.H_init device: {self.H_init.device}")
        # print(f"carry.z_H device: {carry.z_H.device}")
        # print(f"self.L_init device: {self.L_init.device}")
        # print(f"carry.z_L device: {carry.z_L.device}")

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)
        self.carry = None # Will be initialized in forward_loss_wrapper
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
    
    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        batch_device = batch["inputs"].device

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=batch_device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=batch_device),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def _pad_batch(self, batch, pad_size):
        # Convert dtype
        batch = {k: v.to(torch.int32) for k, v in batch.items()}

        # # Convert ignore label IDs
        # if self.metadata.ignore_label_id is not None:
        #     batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        # if batch["puzzle_identifiers"].size < self.local_batch_size:
        # pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
        pad_values = {
            "inputs": PAD_ID,
            "labels": IGNORE_LABEL_ID,
            "puzzle_identifiers": BLANK_IDENTIFIER_ID,
        }

        # Use torch.nn.functional.pad to keep tensors on same device
        padded_batch = {}
        for k, v in batch.items():
            # Build padding tuple: (left, right) for each dimension from last to first
            # We want to pad (0, pad_size) on the first dimension, and (0, 0) on all others
            pad_tuple = (0, 0) * (v.ndim - 1) + (0, pad_size)
            padded_batch[k] = torch.nn.functional.pad(v, pad_tuple, value=pad_values[k])

        return padded_batch

    def valid_forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Always run to max steps during evaluation
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_current_data = batch
        # Run inner model in a loop
        for _ in range(self.config.halt_max_steps):
            new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # Set halted to all true and new steps to max steps
        halted = torch.ones_like(carry.halted, dtype=torch.bool)
        new_steps = self.config.halt_max_steps * torch.ones_like(carry.steps, dtype=torch.int32)

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
    

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

    def forward_loss_wrapper(self, batch, phase):
        # print(batch[0])
        # print(len(batch))
        # print("break")
        # print(batch)
        # breakpoint()
        # print(batch['inputs'].shape)
        # print(batch['labels'].shape)
        # print(batch['puzzle_identifiers'].shape)
        # breakpoint()
        if self.carry is None:
            self.carry = self.initial_carry(batch)

        # Pad batch if it's smaller than carry batch size
        current_batch_size = batch["inputs"].shape[0]
        carry_batch_size = self.carry.halted.shape[0]
        if current_batch_size < carry_batch_size:
            pad_size = carry_batch_size - current_batch_size
            batch = self._pad_batch(batch, pad_size)

        if phase == "train":
            self.carry, outputs = self.forward(self.carry, batch)
            new_carry = self.carry
            labels = new_carry.current_data["labels"]
        elif phase == "valid":
            val_carry = self.initial_carry(batch)
            val_carry, outputs = self.valid_forward(val_carry, batch)
            new_carry = val_carry
            labels = val_carry.current_data["labels"]
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            # breakpoint()
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                # "count": valid_metrics.sum(),
                # "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                # "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
                "count" : valid_metrics.sum() / len(valid_metrics), 
                "accuracy" : (is_correct.to(torch.float32) / loss_divisor).sum(-1).mean(), # Not requiring sequence to be finished for token accuracy
                "exact_accuracy" : (valid_metrics & seq_is_correct).float().mean(),
                "exact_accuracy_unfinished" : seq_is_correct.float().mean(),
            }

        lm_loss = (stablemax_cross_entropy(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum() / carry_batch_size # Average over batch

        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum") / carry_batch_size  # Average over batch

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # breakpoint()
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        return_keys = [] # For now
        # detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        things_to_log = {
            "loss" : lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            # "q_continue_loss": q_continue_loss.detach(),
            "count": metrics["count"].detach(),
            "accuracy": metrics["accuracy"].detach(),
            "exact_accuracy": metrics["exact_accuracy"].detach(),
            "exact_accuracy_unfinished": metrics["exact_accuracy_unfinished"].detach(),
            "steps": metrics["steps"].detach(),
            "q_halt_accuracy": metrics["q_halt_accuracy"].detach(),
        }
        # Add in logging in validation
        if phase == "valid":
            random_idx = torch.randint(0, batch["inputs"].shape[0], (1,)).item()
            grids_to_viz = {
                "input_image": batch["inputs"][random_idx].cpu(),
                "output_image": labels[random_idx].cpu(),
                "pred_image": outputs["preds"][random_idx].cpu(),
            }
            # Create individual visualizations for each grid
            for key, grid in grids_to_viz.items():
                grid_2d = grid.view(30, 30)
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(grid_2d, cmap=self.arc_colormap, vmin=0, vmax=11, interpolation='nearest')
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
                things_to_log[key] = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                plt.close(fig)
            # Create 
        # breakpoint()
        return things_to_log

