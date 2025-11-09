import os
import json
from typing import List
import numpy as np
import pydantic

import torch
from torch.utils.data import Dataset

from model.arc.trm.losses import IGNORE_LABEL_ID
from .common import PuzzleDatasetMetadata

class PuzzleDatasetConfig(pydantic.BaseModel):
    # seed: int
    dataset_paths: List[str]

class PuzzleDataset(Dataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # State
        self._data = None
        self._total_examples = None  # Will be computed after lazy load
        self._index_map = None  # Maps global index to (set_name, local_index)

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

        # Build index map: maps global index to (set_name, local_index)
        self._index_map = []
        self._total_examples = 0
        for set_name, dataset in self._data.items():
            num_examples = len(dataset["inputs"])
            for local_idx in range(num_examples):
                self._index_map.append((set_name, local_idx))
            self._total_examples += num_examples


    def _process_item(self, inputs: np.ndarray, labels: np.ndarray, puzzle_identifier: np.ndarray):
        """Process a single item and convert to tensors."""
        # Convert dtype
        inputs = inputs.astype(np.int32)
        labels = labels.astype(np.int32)
        puzzle_identifier = puzzle_identifier.astype(np.int32)

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            labels = np.where(labels == self.metadata.ignore_label_id, IGNORE_LABEL_ID, labels)

        # Convert to tensors
        return {
            "inputs": torch.from_numpy(inputs),
            "labels": torch.from_numpy(labels),
            "puzzle_identifiers": torch.tensor(puzzle_identifier)
        }
    
    def __len__(self):
        """Return the total number of examples in the dataset."""
        if self._total_examples is None:
            self._lazy_load_dataset()
        return self._total_examples

    def __getitem__(self, idx: int):
        """Get a single item by index."""
        # Lazy load if needed
        if self._data is None:
            self._lazy_load_dataset()

        # Map global index to (set_name, local_index)
        set_name, local_idx = self._index_map[idx]
        dataset = self._data[set_name]

        # Get the puzzle index for this example
        # puzzle_indices contains the start index of each puzzle
        # We need to find which puzzle this example belongs to
        puzzle_idx = np.searchsorted(dataset["puzzle_indices"], local_idx, side="right") - 1

        # Get the data
        inputs = dataset["inputs"][local_idx]
        labels = dataset["labels"][local_idx]
        puzzle_identifier = dataset["puzzle_identifiers"][puzzle_idx]

        # Process and return
        return self._process_item(inputs, labels, puzzle_identifier)

