import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


def _to_numpy(arr):
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


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


def _collect_sixth_image_map(images, texts) -> Dict[str, str]:
    concept_to_images: Dict[str, set] = {}
    for i in range(len(images)):
        concept = _extract_str(texts[i])
        image_key = _extract_str(images[i])
        concept_to_images.setdefault(concept, set()).add(image_key)

    concept_to_sixth = {}
    for concept, image_keys in concept_to_images.items():
        sorted_images = sorted(image_keys)
        if len(sorted_images) < 6:
            continue
        concept_to_sixth[concept] = sorted_images[5]
    return concept_to_sixth


def _build_concept_index(
    images, texts
) -> Tuple[Dict[str, str], Dict[str, List[int]]]:
    concept_to_sixth = _collect_sixth_image_map(images, texts)
    image_to_indices: Dict[str, List[int]] = {}
    for i in range(len(images)):
        image_key = _extract_str(images[i])
        image_to_indices.setdefault(image_key, []).append(i)
    return concept_to_sixth, image_to_indices


def _load_subject_data(train_path: str, subject_id: int) -> Dict:
    loaded = torch.load(train_path, weights_only=False)
    if not (isinstance(loaded, dict) and all(
        key in loaded for key in ["eeg", "label", "img", "text"]
    )):
        raise ValueError(f"Unsupported dataset format in {train_path}.")

    eeg = _to_numpy(loaded["eeg"])
    labels = _to_numpy(loaded["label"])
    images = _to_numpy(loaded["img"])
    texts = _to_numpy(loaded["text"])
    concept_to_image, image_to_indices = _build_concept_index(images, texts)
    if not concept_to_image:
        raise ValueError(f"No concepts with at least 6 images in {train_path}.")

    subject_data = {
        "eeg": eeg,
        "label": labels,
        "img": images,
        "text": texts,
        "concept_to_image": concept_to_image,
        "image_to_indices": image_to_indices,
        "subject_id": subject_id,
        "ch_names": loaded.get("ch_names"),
        "times": loaded.get("times"),
    }
    return subject_data


def _select_concepts(subject_data: List[Dict], seed: int, shuffle: bool) -> List[str]:
    concept_sets = [set(data["concept_to_image"].keys()) for data in subject_data]
    common_concepts = set.intersection(*concept_sets)
    if not common_concepts:
        raise ValueError("No overlapping concepts across subjects.")
    concepts = sorted(common_concepts)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(concepts)
    return concepts


def _build_round_robin_samples(
    subject_data: List[Dict],
    concepts: List[str],
    image_dir: str,
) -> Tuple[Dict, Dict[int, int]]:
    subjects = [data["subject_id"] for data in subject_data]
    subject_lookup = {data["subject_id"]: data for data in subject_data}

    eeg_list = []
    label_list = []
    img_list = []
    text_list = []
    subject_list = []
    subject_counts: Dict[int, int] = {sid: 0 for sid in subjects}

    for i, concept in enumerate(concepts):
        start_idx = i % len(subjects)
        chosen_subject = None
        chosen_index = None
        chosen_image = None
        for offset in range(len(subjects)):
            sid = subjects[(start_idx + offset) % len(subjects)]
            data = subject_lookup[sid]
            image_key = data["concept_to_image"].get(concept)
            if image_key is not None:
                chosen_subject = sid
                chosen_index = data["image_to_indices"].get(image_key)
                chosen_image = image_key
                break
        if chosen_subject is None:
            raise ValueError(f"No subject contains concept {concept}.")

        data = subject_lookup[chosen_subject]
        if not chosen_index:
            raise ValueError(f"No sessions found for concept {concept} in subject {chosen_subject}.")

        image_key = chosen_image
        first_index = chosen_index[0]
        caption = _read_caption(image_dir, image_key, fallback=_extract_str(data["text"][first_index]))

        eeg_stack = np.stack([data["eeg"][idx] for idx in chosen_index], axis=0)
        eeg_avg = eeg_stack.mean(axis=0)

        eeg_list.append(eeg_avg)
        label_list.append(data["label"][first_index])
        img_list.append(data["img"][first_index])
        text_entry = np.array([caption], dtype=object)
        text_list.append(text_entry)
        subject_list.append(chosen_subject)
        subject_counts[chosen_subject] += 1

    output = {
        "eeg": np.stack(eeg_list, axis=0),
        "label": np.stack(label_list, axis=0),
        "img": np.stack(img_list, axis=0),
        "text": np.stack(text_list, axis=0),
        "subject": np.asarray(subject_list, dtype=int),
    }
    return output, subject_counts


def main():
    parser = argparse.ArgumentParser(
        description="Build a test set using the 6th image per concept from THINGS train.pt files."
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
        "--output_path",
        required=True,
        help="Path to save the merged test dataset (.pt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling concepts.",
    )
    parser.add_argument(
        "--shuffle_concepts",
        action="store_true",
        default=True,
        help="Shuffle concepts before round-robin subject assignment.",
    )
    parser.add_argument(
        "--no_shuffle_concepts",
        action="store_false",
        dest="shuffle_concepts",
        help="Disable shuffling concepts (deterministic order).",
    )

    args = parser.parse_args()

    subject_data = []
    for subject_id in args.subjects:
        train_path = os.path.join(
            args.train_root, f"sub-{subject_id:02d}", "train.pt"
        )
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing train.pt for subject {subject_id}: {train_path}")
        subject_data.append(_load_subject_data(train_path, subject_id))

    concepts = _select_concepts(subject_data, seed=args.seed, shuffle=args.shuffle_concepts)
    merged, subject_counts = _build_round_robin_samples(
        subject_data, concepts, args.image_dir
    )

    if subject_data:
        merged["ch_names"] = subject_data[0].get("ch_names")
        merged["times"] = subject_data[0].get("times")

    torch.save(merged, args.output_path)
    counts_msg = ", ".join(
        f"sub-{sid:02d}={count}" for sid, count in sorted(subject_counts.items())
    )
    print(f"Saved {len(concepts)} samples to {args.output_path} ({counts_msg})")


if __name__ == "__main__":
    main()
