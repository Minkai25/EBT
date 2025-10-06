import torch
from einops import rearrange
import torch.nn.functional as F

ARCMAXGRIDSIZE = 30
ARCCHANNELS = 12  # 0-9 + PAD + EOS
def grid_to_seq(grid: torch.Tensor, max_grid_size: int = ARCMAXGRIDSIZE) -> torch.Tensor:
    """
    Convert a grid to a padded grid with EOS markers.
    
    Args:
        grid: Input grid tensor of shape (nrow, ncol) with values 0-9
        max_grid_size: Maximum size for padding
    
    Returns:
        Padded grid tensor of shape (max_grid_size, max_grid_size) as uint8
    """
    
    nrow, ncol = grid.shape
    
    # Shift values: 0-9 becomes 2-11 (leaving 0 for PAD, 1 for EOS)
    grid_shifted = grid + 2
    
    # Pad the grid to max_grid_size x max_grid_size with PAD tokens (0)
    padded_grid = torch.zeros((max_grid_size, max_grid_size), dtype=torch.uint8)
    padded_grid[:nrow, :ncol] = grid_shifted
    
    # Add EOS markers
    # Bottom border (if space available)
    if nrow < max_grid_size:
        padded_grid[nrow, :ncol] = 1
    
    # Right border (if space available)
    if ncol < max_grid_size:
        padded_grid[:nrow, ncol] = 1

    return padded_grid


def one_hot_encode_grids(grids: torch.Tensor, num_channels: int = ARCCHANNELS) -> torch.Tensor:
    """
    One-hot encode grid tensors using einops.
    
    Args:
        grids: Input tensor of shape (H, W) with integer values (any integer dtype)
        num_channels: Number of channels for one-hot encoding (default: 12 for tokens 0-11)
    
    Returns:
        One-hot encoded tensor of shape (num_channels, H, W)
    """
    # Explicit conversion is clearer and avoids warnings
    grids_long = grids.long()

    # One-hot encode: (H, W) -> (H, W, num_channels)
    one_hot = F.one_hot(grids_long, num_classes=num_channels)

    # Rearrange to channel-first: (H, W, C) -> (C, H, W)
    one_hot = rearrange(one_hot, 'h w c -> c h w')

    return one_hot