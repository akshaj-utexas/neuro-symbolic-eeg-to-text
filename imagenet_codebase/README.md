## ImageNet Codebase (SYNAPSE)

This folder contains the ImageNet pipeline used in the paper. Training and inference are driven by `run_pipeline.py`.

### Prerequisites

- Python 3.10+
- Install required Python packages (from your existing environment), at minimum:
  - `torch`, `tqdm`, `pandas`, `nltk`

### Data layout assumptions

`run_pipeline.py` expects:

- EEG dataset `.pth` file (train or test) passed via `--dataset`
- Vocabulary corpus at `--vocab_path` (default: `checkpoints/imagenet_train_corpus.pt`)
- Latent cache created at: `data/imagenet_eeg_test_eeg_latents.pt`

Adjust paths as needed for your local setup.

### Training (SimilarityRefiner)

Trains the same `SimilarityRefiner` model used in inference.

```bash
python run_pipeline.py \
  --mode train \
  --dataset data/imagenet_eeg_train.pth \
  --vocab_path checkpoints/imagenet_train_corpus.pt \
  --epochs 50 \
  --loss focal \
  --batch_size 64 \
  --device cuda
```

The checkpoint is saved under `checkpoints/` (e.g., `similarity_refiner_50eps_focal.pth`).

### Inference

Provide the trained checkpoint path:

```bash
python run_pipeline.py \
  --mode inference \
  --dataset data/imagenet_eeg_test.pth \
  --vocab_path checkpoints/imagenet_train_corpus.pt \
  --checkpoint checkpoints/similarity_refiner_50eps_focal.pth \
  --top_k 15 \
  --batch_size 64 \
  --device cuda
```

This writes alignment outputs into `results/`.

### LLM decoding (optional)

By default, inference continues to LLM-based caption generation. To skip:

```bash
python run_pipeline.py --mode inference --skip_llm ...
```
