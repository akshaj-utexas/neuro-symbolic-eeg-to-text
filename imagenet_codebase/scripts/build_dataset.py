import random
import logging
import torch
from tqdm import tqdm
import json
import os
import numpy as np

from channelnet.datautils import EEGInferenceDataset, SplitterInference

IMAGE_DIR = "data/images"
EEG_DATASET_FILE = "data/eeg_55_95_std.pth"
SPLITS_FILE = "data/block_splits_by_image_all.pth"
OUTPUT_FILE_PATH = "data/eeg_55_95_text_dataset_train.pth"

class Args:
    eeg_dataset = EEG_DATASET_FILE
    image_dir = IMAGE_DIR
    splits_path = SPLITS_FILE
    subject = 0  # all subjects
    time_low = 20
    time_high = 460

args = Args()

base_dataset = EEGInferenceDataset(
    args=args
)
print(f"-----------------------------------------------------\nBase dataset loaded with {len(base_dataset)} samples.")
# 3b. Initialize the Splitter to filter samples and prepare indices.
# We'll use 'all' as the split_name to include all available, filtered data.
final_splitter = SplitterInference(
    dataset=base_dataset,
    split_path=args.splits_path,
    split_num=0,
    split_name="train", # Change from "train" to "all" to get all filtered data
)

print(f"Splitter initialized. Number of samples after filtering: {len(final_splitter)}")

# 3c. Extract the final pairs by iterating through the Splitter

final_eeg_text_dataset = []
print("Starting iteration to extract all EEG-Text pairs...")

for i in tqdm(range(len(final_splitter)), desc="Extracting pairs"):
    # The __getitem__ method of SplitterInference calls the base_dataset __getitem__
    # and returns: eeg, label_string, expected_caption, image_path
    eeg_tensor, object_label, caption_raw, image_path = final_splitter[i]
    
    # Extract the original index from the base_dataset data list
    original_idx_in_base = final_splitter.split_idx[i]
    subject = base_dataset.data[original_idx_in_base]["subject"]
    image_name = base_dataset.images[base_dataset.data[original_idx_in_base]["image"]]

    final_eeg_text_dataset.append(
        {
            "eeg_tensor": eeg_tensor,
            "caption": caption_raw.replace("<s>", "").replace("</s>", ""), # Ensure tokens are stripped
            "object_label": object_label,
            "image_path": image_path,
            "subject": subject,
            "original_eeg_index": original_idx_in_base
        }
    )

# --- 4. VERIFY RESULTS ---
print(f"\nSuccessfully built the dataset with {len(final_eeg_text_dataset)} EEG-Text pairs.")
print("\n--- Example of the first pair ---")
example = final_eeg_text_dataset[0]
print(f"1. EEG Tensor Shape: {example['eeg_tensor'].shape}")
print(f"2. Caption (Text): {example['caption']}")
print(f"3. Object Label: {example['object_label']}")
print(f"4. Subject: {example['subject']}")
print(f"5. Image Path: {example['image_path']}")

try:
    print(f"\nSaving final dataset to {OUTPUT_FILE_PATH}...")
    torch.save(final_eeg_text_dataset, OUTPUT_FILE_PATH)
    print("Dataset successfully saved!")
except Exception as e:
    print(f"An error occurred while saving the dataset: {e}")