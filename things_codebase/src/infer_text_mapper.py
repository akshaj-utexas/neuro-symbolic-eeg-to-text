import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from safetensors.torch import load_file

from args import get_args_for_text_mapper_inference
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from models import SimilarityRefiner


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


def _extract_item(arr, index):
    item = arr[index]
    if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0:
        item = item[0]
    return item


def _resolve_image_path(image_dir, image_key):
    if os.path.isabs(image_key):
        return image_key
    primary = os.path.join(image_dir, image_key)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(image_dir, "Image_set", image_key)
    if os.path.exists(fallback):
        return fallback
    return primary


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


def _load_subject_data(test_path, time_low, time_high, image_dir):
    loaded = torch.load(test_path, weights_only=False)
    if not (isinstance(loaded, dict) and all(k in loaded for k in ["eeg", "img", "text"])):
        raise ValueError(f"Unsupported test dataset format in {test_path}.")

    eeg = loaded["eeg"]
    imgs = loaded["img"]
    texts = loaded["text"]

    if isinstance(eeg, np.ndarray):
        eeg = torch.from_numpy(eeg)
    if eeg.dim() == 4:
        eeg = eeg.mean(dim=1)

    groups = {}
    for i in range(eeg.shape[0]):
        image_key = str(_extract_item(imgs, i))
        caption = str(_extract_item(texts, i))
        eeg_i = eeg[i].float()
        groups.setdefault(image_key, {"caption": caption, "eegs": []})
        groups[image_key]["eegs"].append(eeg_i)

    items = []
    for image_key, payload in groups.items():
        stacked = torch.stack(payload["eegs"])
        avg = stacked.mean(dim=0)
        avg = avg[:, int(time_low) : int(time_high)]
        avg = avg.unsqueeze(0)  # [1, C, T]
        caption = _read_caption(image_dir, image_key, payload["caption"])
        items.append(
            {
                "eeg": avg,
                "caption": caption,
                "image": image_key,
            }
        )

    return items


class AveragedEEGDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item["eeg"], item["caption"], item["image"]


class MergedEEGDataset(Dataset):
    def __init__(self, loaded, time_low, time_high):
        self.time_low = int(time_low)
        self.time_high = int(time_high)
        self.eeg = loaded["eeg"]
        self.images = loaded["img"]
        self.texts = loaded["text"]
        self.subjects = loaded.get("subject")

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        if eeg.dim() == 3:
            eeg = eeg.mean(dim=0)
        eeg = eeg.float()
        eeg = eeg[:, self.time_low : self.time_high]
        eeg = eeg.unsqueeze(0)

        image_key = str(_extract_item(self.images, idx))
        caption = str(_extract_item(self.texts, idx))
        subject = None
        if self.subjects is not None:
            subject = int(_extract_item(self.subjects, idx))
        return eeg, caption, image_key, subject, idx


def main():
    args = get_args_for_text_mapper_inference()
    if args.test_path:
        test_path = args.test_path
        output_path = args.output_path or test_path
    else:
        if not args.test_root:
            raise ValueError("Provide --test_path or --test_root.")
        if not args.image_dir:
            raise ValueError("Provide --image_dir when using --test_root.")
        os.makedirs(args.output_dir, exist_ok=True)

    text_mapper_ckpt = torch.load(args.text_mapper_ckpt, map_location="cpu")
    vocab_corpus = args.vocab_corpus or text_mapper_ckpt.get("vocab_corpus")
    if not vocab_corpus:
        raise ValueError("Provide --vocab_corpus or use a checkpoint with vocab_corpus.")

    vocab_info = torch.load(vocab_corpus, map_location="cpu")
    hidden_dim = text_mapper_ckpt.get("hidden_dim", 1024)
    use_scaling = text_mapper_ckpt.get("use_scaling", False)
    refiner = SimilarityRefiner(
        vocab_info["embeddings"],
        input_dim=512,
        hidden_dim=hidden_dim,
        use_scaling=use_scaling,
    )
    refiner.load_state_dict(text_mapper_ckpt["model_state_dict"], strict=False)
    refiner.to(args.device).eval()

    vocab_words = vocab_info["words"]

    if args.test_path:
        loaded = torch.load(test_path, weights_only=False)
        if not isinstance(loaded, dict):
            raise ValueError(f"Unsupported test dataset format in {test_path}.")

        if args.use_precomputed_latents:
            if "eeg_clip_latent" not in loaded:
                raise ValueError("Missing eeg_clip_latent in test file for latent-only inference.")

            latents = loaded["eeg_clip_latent"]
            if torch.is_tensor(latents):
                latents = latents.cpu()
            elif isinstance(latents, np.ndarray):
                latents = torch.from_numpy(latents)
            else:
                latents = torch.stack([torch.as_tensor(x) for x in latents], dim=0)

            if latents.dim() == 3 and latents.size(1) == 1:
                latents = latents.squeeze(1)
            if latents.dim() == 1:
                latents = latents.unsqueeze(0)

            total = latents.size(0)
            all_top_words = [None] * total
            all_refined_latents = torch.zeros(
                (total, latents.size(-1)), dtype=latents.dtype
            )

            with torch.no_grad():
                for start in tqdm(range(0, total, args.batch_size), desc="Latent-only"):
                    batch = latents[start : start + args.batch_size].to(args.device).float()
                    logits, refined = refiner(batch)
                    scores, top_indices = logits.topk(args.top_k, dim=-1)

                    for i in range(batch.size(0)):
                        row_idx = start + i
                        top_words = [
                            {"word": vocab_words[idx], "score": float(score)}
                            for idx, score in zip(
                                top_indices[i].tolist(), scores[i].tolist()
                            )
                        ]
                        all_top_words[row_idx] = top_words
                        all_refined_latents[row_idx] = refined[i].cpu()

            loaded["top_words"] = all_top_words
            loaded["refined_latent"] = all_refined_latents
            torch.save(loaded, output_path)
            print(f"Appended latent-only results to {output_path}")
            return

        if not all(k in loaded for k in ["eeg", "img", "text"]):
            raise ValueError(f"Unsupported test dataset format in {test_path}.")

        dataset = MergedEEGDataset(loaded, args.time_low, args.time_high)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        sample_eeg = dataset[0][0].squeeze(0)
        config = _resolve_config(args.encoder_path, args.time_low, args.time_high, sample_eeg)
        encoder = ChannelNetModel(config=config)
        _load_weights(encoder, args.encoder_path)
        encoder.to(args.device).eval()

        total = len(dataset)
        all_top_words = [None] * total
        all_eeg_latents = None
        all_refined_latents = None

        with torch.no_grad():
            for eeg, captions, images, subjects, indices in tqdm(
                loader, desc="Merged test"
            ):
                eeg = eeg.to(args.device)
                embeddings, _ = encoder(eeg)
                logits, refined = refiner(embeddings)
                scores, top_indices = logits.topk(args.top_k, dim=-1)

                if all_eeg_latents is None:
                    all_eeg_latents = torch.zeros(
                        (total, embeddings.shape[-1]), dtype=embeddings.dtype
                    )
                    all_refined_latents = torch.zeros(
                        (total, refined.shape[-1]), dtype=refined.dtype
                    )

                for i in range(embeddings.size(0)):
                    row_idx = int(indices[i])
                    top_words = [
                        {"word": vocab_words[idx], "score": float(score)}
                        for idx, score in zip(top_indices[i].tolist(), scores[i].tolist())
                    ]
                    all_top_words[row_idx] = top_words
                    all_eeg_latents[row_idx] = embeddings[i].cpu()
                    all_refined_latents[row_idx] = refined[i].cpu()

        loaded["top_words"] = all_top_words
        loaded["eeg_clip_latent"] = all_eeg_latents
        loaded["refined_latent"] = all_refined_latents
        torch.save(loaded, output_path)
        print(f"Appended inference results to {output_path}")
        return

    for subject_id in args.subjects:
        test_path = os.path.join(args.test_root, f"sub-{subject_id:02d}", "test.pt")
        items = _load_subject_data(test_path, args.time_low, args.time_high, args.image_dir)
        if not items:
            raise ValueError(f"No samples found for subject {subject_id} at {test_path}.")

        dataset = AveragedEEGDataset(items)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        sample_eeg = items[0]["eeg"].squeeze(0)
        config = _resolve_config(args.encoder_path, args.time_low, args.time_high, sample_eeg)
        encoder = ChannelNetModel(config=config)
        _load_weights(encoder, args.encoder_path)
        encoder.to(args.device).eval()

        results = []
        with torch.no_grad():
            for eeg, captions, images in tqdm(
                loader, desc=f"Subject {subject_id:02d}"
            ):
                eeg = eeg.to(args.device)
                embeddings, _ = encoder(eeg)
                logits, refined = refiner(embeddings)
                scores, indices = logits.topk(args.top_k, dim=-1)

                for i in range(embeddings.size(0)):
                    top_words = [
                        {"word": vocab_words[idx], "score": float(score)}
                        for idx, score in zip(indices[i].tolist(), scores[i].tolist())
                    ]
                    results.append(
                        {
                            "subject": subject_id,
                            "image": images[i],
                            "caption": captions[i],
                            "eeg_clip_latent": embeddings[i].cpu().unsqueeze(0),
                            "refined_latent": refined[i].cpu().unsqueeze(0),
                            "top_words": top_words,
                        }
                    )

        output_path = os.path.join(
            args.output_dir, f"subject_{subject_id:02d}_top_words_training_split.pt"
        )
        torch.save(results, output_path)
        print(f"Saved {len(results)} samples to {output_path}")


if __name__ == "__main__":
    main()
