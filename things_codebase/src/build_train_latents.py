import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import load_file

from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig


def _to_numpy(arr):
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


def _resolve_config(encoder_path, time_low, time_high, sample):
    try:
        return EEGModelConfig.from_pretrained(encoder_path)
    except Exception:
        input_height = sample.shape[0]
        input_width = max(1, int(time_high) - int(time_low))
        return EEGModelConfig(input_height=input_height, input_width=input_width)


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


def _resolve_image_path(image_dir, image_key):
    if os.path.isabs(image_key):
        return image_key
    candidates = []
    candidates.append(os.path.join(image_dir, image_key))
    candidates.append(os.path.join(image_dir, "Image_set", image_key))

    normalized = image_key.replace("\\", "/")
    if "Image_set/" in normalized:
        normalized = normalized.split("Image_set/", 1)[1]
    if "train_images/" in normalized:
        normalized = normalized.split("train_images/", 1)[1]
    if "training_images/" in normalized:
        normalized = normalized.split("training_images/", 1)[1]
    if normalized != image_key:
        candidates.append(os.path.join(image_dir, normalized))

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _read_caption(image_dir, image_key, fallback):
    image_path = _resolve_image_path(image_dir, image_key)
    if os.path.isdir(image_path):
        txt_files = sorted(
            [
                os.path.join(image_path, fname)
                for fname in os.listdir(image_path)
                if fname.lower().endswith(".txt")
            ]
        )
        if txt_files:
            with open(txt_files[0], "r", encoding="utf-8") as handle:
                return handle.readline().strip()
        return fallback

    base, _ = os.path.splitext(image_path)
    caption_path = f"{base}.txt"
    if not os.path.exists(caption_path):
        alt = image_path
        alt = alt.replace(".JPEG", ".txt").replace(".jpg", ".txt").replace(".png", ".txt")
        if alt != caption_path and os.path.exists(alt):
            caption_path = alt
        else:
            return fallback
    with open(caption_path, "r", encoding="utf-8") as handle:
        return handle.readline().strip()


def _extract_str(item):
    if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0:
        return str(item[0])
    return str(item)


def _collect_first_five_images(images, texts) -> Dict[str, List[str]]:
    concept_to_images: Dict[str, set] = {}
    for i in range(len(images)):
        concept = _extract_str(texts[i])
        image_key = _extract_str(images[i])
        concept_to_images.setdefault(concept, set()).add(image_key)

    concept_to_first_five = {}
    for concept, image_keys in concept_to_images.items():
        sorted_images = sorted(image_keys)
        concept_to_first_five[concept] = sorted_images[:5]
    return concept_to_first_five


def _normalize_eeg_sample(sample):
    if sample.ndim == 3:
        return sample.mean(axis=0)
    return sample


def _load_subject_entries(train_path: str, subject_id: int, image_dir: str):
    loaded = torch.load(train_path, weights_only=False)
    if not (isinstance(loaded, dict) and all(
        key in loaded for key in ["eeg", "label", "img", "text"]
    )):
        raise ValueError(f"Unsupported dataset format in {train_path}.")

    eeg = _to_numpy(loaded["eeg"])
    labels = _to_numpy(loaded["label"])
    images = _to_numpy(loaded["img"])
    texts = _to_numpy(loaded["text"])

    concept_to_images = _collect_first_five_images(images, texts)
    image_to_indices: Dict[str, List[int]] = {}
    for i in range(len(images)):
        image_key = _extract_str(images[i])
        image_to_indices.setdefault(image_key, []).append(i)

    entries = []
    for concept_images in concept_to_images.values():
        for image_key in concept_images:
            indices = image_to_indices.get(image_key, [])
            if not indices:
                continue
            eeg_stack = np.stack(
                [_normalize_eeg_sample(eeg[idx]) for idx in indices], axis=0
            )
            eeg_avg = eeg_stack.mean(axis=0)
            first_index = indices[0]
            caption = _read_caption(
                image_dir, image_key, fallback=_extract_str(texts[first_index])
            )
            entries.append(
                {
                    "eeg": eeg_avg,
                    "label": labels[first_index],
                    "img": images[first_index],
                    "caption": caption,
                    "subject": subject_id,
                }
            )

    return entries, loaded.get("ch_names"), loaded.get("times")


def main():
    parser = argparse.ArgumentParser(
        description="Build averaged train EEG latents (images 1-5) for exemplar retrieval."
    )
    parser.add_argument(
        "--train_root",
        required=True,
        help="Path to Preprocessed_data_250Hz_whiten (contains sub-*/train.pt).",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Subject IDs to include.",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Root directory containing the image files and captions.",
    )
    parser.add_argument(
        "--encoder_path",
        required=True,
        help="Path to the trained ChannelNet encoder checkpoint directory.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the merged training latent dataset (.pt).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )

    args = parser.parse_args()

    all_entries = []
    ch_names = None
    times = None
    for subject_id in args.subjects:
        train_path = os.path.join(
            args.train_root, f"sub-{subject_id:02d}", "train.pt"
        )
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing train.pt for subject {subject_id}: {train_path}")
        entries, subject_ch_names, subject_times = _load_subject_entries(
            train_path, subject_id, args.image_dir
        )
        all_entries.extend(entries)
        if ch_names is None:
            ch_names = subject_ch_names
        if times is None:
            times = subject_times

    if not all_entries:
        raise ValueError("No training entries found.")

    sample_eeg = all_entries[0]["eeg"]
    config = _resolve_config(args.encoder_path, args.time_low, args.time_high, sample_eeg)
    encoder = ChannelNetModel(config=config)
    _load_weights(encoder, args.encoder_path)
    encoder.to(args.device).eval()

    total = len(all_entries)
    eeg_latents = torch.zeros((total, config.embedding_size), dtype=torch.float32)

    with torch.no_grad():
        for start in tqdm(range(0, total, args.batch_size), desc="Encoding train latents"):
            batch_entries = all_entries[start : start + args.batch_size]
            eeg_batch = np.stack([entry["eeg"] for entry in batch_entries], axis=0)
            eeg_batch = eeg_batch[:, :, int(args.time_low) : int(args.time_high)]
            eeg_batch = torch.from_numpy(eeg_batch).float().unsqueeze(1)
            eeg_batch = eeg_batch.to(args.device)
            embeddings, _ = encoder(eeg_batch)
            eeg_latents[start : start + len(batch_entries)] = embeddings.cpu()

    output = {
        "eeg": np.stack([entry["eeg"] for entry in all_entries], axis=0),
        "label": np.stack([entry["label"] for entry in all_entries], axis=0),
        "img": np.stack([entry["img"] for entry in all_entries], axis=0),
        "caption": np.array([entry["caption"] for entry in all_entries], dtype=object),
        "subject": np.asarray([entry["subject"] for entry in all_entries], dtype=int),
        "eeg_clip_latent": eeg_latents,
    }
    if ch_names is not None:
        output["ch_names"] = ch_names
    if times is not None:
        output["times"] = times

    torch.save(output, args.output_path)
    print(f"Saved {total} training latents to {args.output_path}")


if __name__ == "__main__":
    main()
