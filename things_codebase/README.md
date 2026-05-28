# THINGS EEG Pipeline

This codebase runs the THINGS EEG pipeline end‑to‑end: train a ChannelNet EEG encoder, train a multi‑hot EEG→CLIP text mapper, and run inference with GraphRAG pruning and LLM captioning + metrics.

## Dataset format (THINGS EEG)

The loader expects `.pt` files containing:

- `eeg`: shape `(N, 4, C, T)` (4 repeats per image)
- `label`: shape `(N, 4)`
- `img`: shape `(N, 4)` with relative image paths
- `text`: shape `(N, 4)` with class names
- `ch_names`, `times`, `session` (optional metadata)

## Codebase structure

Top‑level entry point: `run_pipeline.py` for text‑mapper training + inference.

`src/` highlights:

- **ChannelNet encoder**: `train_eeg_classifier.py`, `test_eeg_classifier.py`, `encoders.py`
- **Latent generation**: `encode_eeg_latents.py`, `build_test_from_train.py`, `build_train_latents.py`
- **Text mapper**: `train_text_mapper.py`, `infer_text_mapper.py`, `models.py`, `trainer.py`, `loss.py`
- **GraphRAG + exemplars**: `graph_filter.py`, `exemplar_retriever.py`, `prune_subject_bow.py`
- **LLM + metrics**: `run_pruned_bow_llm.py`, `llm_client.py`, `metrics.py`, `llm_eval.py`
- **Shared utilities**: `datautils.py`, `constants.py`, `args.py`

### What changed
- **THINGS EEG format support** in `datautils.EEGDataset`.
- **4 repeats are averaged per subject** before feeding the encoder.
- **Only the first 5 images per class** are used (by filename order within each class).
- **Optional subject limiting** via `--max_subjects`.
- **Input height/width are auto‑configured** from the dataset (`C` channels and `T` timepoints).
- **Separate train/val/test dataset args** are supported.

---

## Commands

Use placeholders like `/path/to/...` or `$DATA_ROOT` and `$OUTPUT_DIR` to match your environment.

### One entry point (MLP pipeline)

Use the top‑level pipeline for **text‑mapper training** and **inference + GraphRAG + LLM**:

```bash
python run_pipeline.py --mode train ...
python run_pipeline.py --mode inference ...
```

ChannelNet encoder training remains a separate step (below).

### 1) Train the EEG encoder (limit to 5 subjects)
```bash
python src/train_eeg_classifier.py \
  --eeg_train_dataset /path/to/things-eeg/Preprocessed_data_250Hz_whiten \
  --eeg_test_dataset  /path/to/things-eeg/Preprocessed_data_250Hz_whiten \
  --image_dir /path/to/things-eeg/Image_set \
  --time_low 0 --time_high 250 \
  --max_subjects 5 \
  --output /path/to/output/things-eeg-all
```

### 2) (Optional) Evaluate a trained checkpoint (all subjects)
```bash
python src/test_eeg_classifier.py \
  --eeg_test_dataset /path/to/things-eeg/Preprocessed_data_250Hz_whiten \
  --image_dir /path/to/things-eeg/Image_set \
  --time_low 0 --time_high 250 \
  --output /path/to/output/things-eeg-all
```

### 3) Encode EEG latents for the mapper
```bash
python src/encode_eeg_latents.py \
  --encoder_path /path/to/output/things-eeg-all \
  --eeg_train_dataset /path/to/things-eeg/Preprocessed_data_250Hz_whiten \
  --time_low 0 --time_high 250 \
  --max_subjects 5 \
  --output_path /path/to/output/things-eeg-all/encoded_eeg_train.pth
```

### 4) Train EEG-to-CLIP text mapper (multi-hot, focal loss only)
```bash
python run_pipeline.py --mode train \
  --encoded_eeg_dataset /path/to/output/things-eeg-all/encoded_eeg_train.pth \
  --vocab_corpus /path/to/data/things_train_corpus.pt \
  --output /path/to/output/text-mapper \
  --num_epochs 5 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --use_scaling \
  --focal_alpha 0.25 \
  --focal_gamma 2.0
```

### 5) Build merged test.pt (6th image, averaged sessions)
```bash
python src/build_test_from_train.py \
  --train_root /path/to/things-eeg/Preprocessed_data_250Hz_whiten \
  --image_dir /path/to/Image_set/training_images \
  --output_path /path/to/things-eeg/test_from_train.pt \
  --subjects 1 2 3 4 5
```

### 6) Inference: merged test.pt → top-15 words → GraphRAG → LLM + metrics
```bash
python run_pipeline.py --mode inference \
  --test_path /path/to/things-eeg/test_from_train.pt \
  --encoder_path /path/to/output/things-eeg-all \
  --text_mapper_ckpt /path/to/output/text-mapper/text_mapper_best.pt \
  --train_latents_path /path/to/things-eeg/train_latents.pt \
  --vocab_corpus /path/to/data/things_train_corpus.pt \
  --time_low 0 --time_high 250 \
  --batch_size 128 \
  --top_k 15 \
  --model chatgpt
```

### 7) Build training latents (images 1–5, averaged sessions)
```bash
python src/build_train_latents.py \
  --train_root /path/to/things-eeg/Preprocessed_data_250Hz_whiten \
  --image_dir /path/to/Image_set/training_images \
  --encoder_path /path/to/output/things-eeg-all \
  --output_path /path/to/things-eeg/train_latents.pt \
  --subjects 1 2 3 4 5
```

### 8) (Advanced) Run pruning or LLM steps manually
If you want to run the post‑processing stages independently, use:
`src/prune_subject_bow.py` and `src/run_pruned_bow_llm.py`.

---

## Notes
- `--image_dir` should be the root that contains `training_images/...` and `test_images/...` so that paths in `img` can be joined directly.
- The model auto‑sets `input_height` to `C` and `input_width` to `time_high - time_low`.
- If `--eeg_train_dataset`/`--eeg_test_dataset` points to a directory, it automatically loads and concatenates all `sub-*/train.pt` or `sub-*/test.pt` files. Use `--max_subjects` to cap the number of subjects.

---

## Script reference (updated)
| Script | Purpose |
| --- | --- |
| `src/build_test_from_train.py` | Build `test_from_train.pt` from the 6th image per concept (averaged sessions, round‑robin across subjects). |
| `src/infer_text_mapper.py` | Append `top_words`, `eeg_clip_latent`, and `refined_latent` into the merged test file. |
| `src/build_train_latents.py` | Build `train_latents.pt` from images 1–5 per concept (averaged sessions + ChannelNet latents). |
| `src/prune_subject_bow.py` | Prune top‑K words via ConceptNet and attach exemplar captions using `train_latents.pt`. |
