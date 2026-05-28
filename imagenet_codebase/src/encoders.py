import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
from torch.utils.data import Dataset, DataLoader

import json
from safetensors.torch import load_file 

MODEL_REGISTRY = {
    "channelnet": {
        "config": "models/config.json",
        "model": "models/model.safetensors"
    }
    # Add more encoders later
}
DATASET_REGISTRY = {
    "imagenet_eeg": "data/imagenet_eeg_text_dataset_all_subjects_5_95.pth",
    "imagenet_eeg_test": "data/eeg_55_95_text_dataset_test.pth",
    "imagenet_eeg_train": "data/eeg_55_95_text_dataset_train.pth"
}

sys.path.append(os.getcwd())
# Thought2Text ChannelNet EEG Encoder

from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from channelnet.constants import id2label
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
# --- TRANSFORMERS COMPATIBILITY PATCH ---
# Newer versions of transformers expect 'all_tied_weights_keys'
# Force all_tied_weights_keys to always be a dict to prevent NoneType errors in transformers v4.36+
def _get_tied_keys(self):
    return {}

ChannelNetModel.all_tied_weights_keys = property(_get_tied_keys)
# ----------------------------------------
class PreprocessedEEGDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        # Returns [1, 128, 440] tensor and metadata
        return item['eeg_tensor'], item['caption'], item['object_label'], item['subject']

@torch.no_grad()
def process_channelnet(dataset_path, output_path, device, batch_size):
    """
    Inference function for ChannelNet to extract CLIP-aligned latents.
    """
    config_path = MODEL_REGISTRY["channelnet"]["config"]
    model_path = MODEL_REGISTRY["channelnet"]["model"]
    # 1. Load Model & Config
    config = EEGModelConfig.from_json_file(config_path)
    model = ChannelNetModel.from_pretrained(model_path, config=config)
    model.to(device).eval()
    
    dataset = DATASET_REGISTRY[dataset_path]

    # 2. Prepare Data
    ds = PreprocessedEEGDataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    encoded_dataset = []
    print(f"Generating CLIP latents for {len(ds)} samples...")

    for eeg_tensors, captions, labels, subjects in tqdm(loader, desc="ChannelNet Encoding"):
        eeg_tensors = eeg_tensors.to(device)
        
        # Extract 512-D CLIP-aligned embedding (H_eeg) 
        # ChannelNet returns (embedding, classification_logits)
        embeddings, logits = model(eeg_tensors)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        
        for i in range(embeddings.size(0)):

            pred_idx_str = str(preds[i].item())
            pred_label = id2label.get(pred_idx_str, "unknown")

            encoded_dataset.append({
                "eeg_clip_latent": embeddings[i].cpu().unsqueeze(0),  # [512] -> [1, 512]
                "predicted_object_label": pred_label,
                "prediction_confidence": confidences[i].item(),
                "caption": captions[i],
                "object_label": labels[i],
                "subject": subjects[i].item()
            })

    torch.save(encoded_dataset, output_path)
    print(f"Saved {len(encoded_dataset)} encoded samples to {output_path}")