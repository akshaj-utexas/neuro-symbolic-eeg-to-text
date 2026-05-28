import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

from args import get_args_for_latent_encoding
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from datautils import EEGDataset, Splitter, EncodedEEGDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_config(encoder_path, base_dataset):
    try:
        return EEGModelConfig.from_pretrained(encoder_path)
    except Exception:
        input_height = getattr(base_dataset, "num_channels", 128)
        time_low = max(0, int(base_dataset.args.time_low))
        time_high = min(
            getattr(base_dataset, "num_timepoints", base_dataset.args.time_high),
            int(base_dataset.args.time_high),
        )
        input_width = max(1, time_high - time_low)
        if hasattr(base_dataset, "labels"):
            labels = base_dataset.labels
            if isinstance(labels, torch.Tensor):
                max_label = int(labels.max().item())
            else:
                labels = np.asarray(labels)
                max_label = int(labels.max())
            num_classes = max_label + 1
        else:
            num_classes = EEGModelConfig().num_classes
        return EEGModelConfig(
            input_height=input_height, input_width=input_width, num_classes=num_classes
        )


def _load_weights(model, encoder_path):
    if os.path.isfile(encoder_path):
        weight_path = encoder_path
    else:
        safetensors_path = os.path.join(encoder_path, "model.safetensors")
        bin_path = os.path.join(encoder_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            weight_path = safetensors_path
        elif os.path.exists(bin_path):
            weight_path = bin_path
        else:
            raise FileNotFoundError(
                f"No model weights found under {encoder_path} (expected model.safetensors or pytorch_model.bin)."
            )

    if weight_path.endswith(".safetensors"):
        state_dict = load_file(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    args = get_args_for_latent_encoding()
    set_seed(42)

    if args.eeg_train_dataset:
        base_dataset = EEGDataset(
            args=args,
            eeg_dataset_path=args.eeg_train_dataset,
            dataset_split="train",
        )
        indices = None
    else:
        if args.eeg_dataset is None or args.splits_path is None:
            raise ValueError(
                "Provide --eeg_train_dataset or --eeg_dataset with --splits_path."
            )
        base_dataset = EEGDataset(args=args)
        if getattr(base_dataset, "format", None) == "things":
            raise ValueError(
                "THINGS datasets should be passed via --eeg_train_dataset (train.pt)."
            )
        split = Splitter(
            base_dataset,
            split_path=args.splits_path,
            split_num=args.split_num,
            split_name="train",
        )
        indices = split.split_idx

    dataset = EncodedEEGDataset(base_dataset, indices=indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    config = _resolve_config(args.encoder_path, base_dataset)
    model = ChannelNetModel(config=config)
    _load_weights(model, args.encoder_path)
    model.to(args.device).eval()

    encoded = []
    with torch.no_grad():
        for eeg, captions, labels in tqdm(loader, desc="Encoding EEG latents"):
            eeg = eeg.to(args.device)
            embeddings, _ = model(eeg)
            for i in range(embeddings.size(0)):
                encoded.append(
                    {
                        "eeg_clip_latent": embeddings[i].cpu().unsqueeze(0),
                        "caption": captions[i],
                        "label": int(labels[i]),
                    }
                )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(encoded, args.output_path)
    print(f"Saved {len(encoded)} encoded samples to {args.output_path}")


if __name__ == "__main__":
    main()
