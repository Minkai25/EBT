from torch.utils.data import Dataset
from typing import Tuple, List
import json
import os
import torch
from typing import Set
import numpy as np
from .arc_utils import grid_to_seq, one_hot_encode_grids


class ARCDataset_no_post_train(Dataset):
    """Base dataset for ARC examples (no post-training)"""

    def __init__(self, 
                 data_dirs: List[str] = None,
                 split: str = 'train',
                 augmentation_factor: int = 0, 
                 max_retries_per_example: int = 0):
        """
        Args:
            data_dirs: List of directories containing JSON files. If None, uses default training dir.
            split: Either 'train' or 'val' to determine which examples to load
            augmentation_factor: Number of augmented versions per original example
            max_retries_per_example: Maximum attempts to generate unique augmentations per example
        """
        if data_dirs is None:
            data_dirs = ["data/arc/raw-data/ARC-AGI/data/training"]
        
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.split = split
        self.augmentation_factor = augmentation_factor
        self.max_retries_per_example = max_retries_per_example
        self.examples = []
        self.seen_hashes: Set[str] = set()
        
        # Collect all JSON files from all directories
        all_json_files = []
        for data_dir in self.data_dirs:
            json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            json_files = [os.path.join(data_dir, f) for f in json_files]
            all_json_files.extend(json_files)
        
        all_json_files = sorted(all_json_files)  # Ensure consistent ordering
        print(f"Using {len(all_json_files)} files for {split} dataset")

        for file_idx, json_file in enumerate(all_json_files):
            self._load_file(json_file, file_idx)

        print(f"Loaded {len(self.examples)} examples for {split} split")

    def _load_file(self, json_file: str, file_idx: int):
        """Load a single JSON file and add examples from the specified split."""
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Determine which examples to load based on split
        if self.split == 'train':
            examples_to_load = data['train']
        elif self.split == 'val':
            examples_to_load = data['test']
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")

        for example_idx, example in enumerate(examples_to_load):
            input_tensor = torch.tensor(example['input'], dtype=torch.uint8)
            output_tensor = torch.tensor(example['output'], dtype=torch.uint8)
            input_tensor = one_hot_encode_grids(grid_to_seq(input_tensor))
            output_tensor = one_hot_encode_grids(grid_to_seq(output_tensor))
            
            # Add original example
            original_example = (input_tensor, output_tensor, torch.tensor(file_idx), False)  # False = not augmented
            self.examples.append(original_example)

            # # Generate augmented versions
            # augmented_count = 0
            # for trial in range(self.max_retries_per_example):
            #     if augmented_count >= self.augmentation_factor:
            #         break

            #     # Generate color mapping (preserve 0, permute 1-9)
            #     color_mapping = self._generate_color_mapping()

            #     # Apply color mapping
            #     aug_input = self._apply_color_mapping(input_tensor, color_mapping)
            #     aug_output = self._apply_color_mapping(output_tensor, color_mapping)

            #     # Check for duplicates
            #     aug_hash = self._compute_hash(aug_input, aug_output)

            #     if aug_hash not in self.seen_hashes:
            #         self.seen_hashes.add(aug_hash)
            #         aug_example = (aug_input, aug_output, torch.tensor(file_idx), True)  # True = augmented
            #         self.examples.append(aug_example)
            #         augmented_count += 1

    # def _generate_color_mapping(self) -> np.ndarray:
    #     """Generate a color mapping that preserves 0 (black) and permutes colors 1-9."""
    #     mapping = np.concatenate([
    #         np.array([0], dtype=np.uint8),  # Keep black (0) unchanged
    #         np.random.permutation(np.arange(1, 10, dtype=np.uint8))  # Shuffle colors 1-9
    #     ])
    #     return mapping

    # def _apply_color_mapping(self, tensor: torch.Tensor, mapping: np.ndarray) -> torch.Tensor:
    #     """Apply color mapping to a tensor."""
    #     # Convert to numpy for mapping, then back to tensor
    #     numpy_array = tensor.cpu().numpy()
    #     mapped_array = mapping[numpy_array]
    #     return torch.tensor(mapped_array, dtype=torch.long, device=self.device)

    # def _compute_hash(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> str:
    #     """Compute hash of an input-output pair to detect duplicates."""
    #     # Convert tensors to bytes for hashing
    #     input_bytes = input_tensor.cpu().numpy().tobytes()
    #     output_bytes = output_tensor.cpu().numpy().tobytes()

    #     # Create combined hash
    #     combined = input_bytes + output_bytes
    #     return hashlib.md5(combined).hexdigest()

    # def _count_augmented(self) -> int:
    #     """Count number of augmented examples."""
    #     return sum(1 for _, _, _, is_augmented in self.examples if is_augmented)

    # def get_augmentation_stats(self) -> Dict[str, int]:
    #     """Get statistics about original vs augmented examples."""
    #     original_count = sum(1 for _, _, _, is_augmented in self.examples if not is_augmented)
    #     augmented_count = sum(1 for _, _, _, is_augmented in self.examples if is_augmented)

    #     return {
    #         'original_examples': original_count,
    #         'augmented_examples': augmented_count,
    #         'total_examples': len(self.examples),
    #         'augmentation_ratio': augmented_count / original_count if original_count > 0 else 0
    #     }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return input, output, and file_index tensors (excluding augmentation flag)."""
        input_tensor, output_tensor, file_idx, _ = self.examples[idx]
        return input_tensor, output_tensor, file_idx


class ARCTrainDataset_no_post_train(ARCDataset_no_post_train):
    """Dataset for ARC training examples (no post-training)"""
    
    def __init__(self, 
                 data_dirs: List[str] = None,
                 augmentation_factor: int = 0, 
                 max_retries_per_example: int = 0):
        """
        Args:
            data_dirs: List of directories containing JSON files. If None, uses default training dir.
            augmentation_factor: Number of augmented versions per original example
            max_retries_per_example: Maximum attempts to generate unique augmentations per example
        """
        super().__init__(
            data_dirs=data_dirs,
            split='train',
            augmentation_factor=augmentation_factor,
            max_retries_per_example=max_retries_per_example
        )


class ARCValDataset_no_post_train(ARCDataset_no_post_train):
    """Dataset for ARC validation examples (no post-training)"""
    
    def __init__(self, 
                 data_dirs: List[str] = None,
                 augmentation_factor: int = 0, 
                 max_retries_per_example: int = 0):
        """
        Args:
            data_dirs: List of directories containing JSON files. If None, uses default training dir.
            augmentation_factor: Number of augmented versions per original example
            max_retries_per_example: Maximum attempts to generate unique augmentations per example
        """
        super().__init__(
            data_dirs=data_dirs,
            split='val',
            augmentation_factor=augmentation_factor,
            max_retries_per_example=max_retries_per_example
        )