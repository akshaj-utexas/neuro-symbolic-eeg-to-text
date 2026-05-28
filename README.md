# SYNAPSE EEG Pipelines

> Parallel codebases for EEG-to-text decoding and analysis as featured in the SYNAPSE paper.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/pytorch-v2.0%2B-ee4c2c.svg)

This repository contains two self-contained, parallel codebases used to process, train, and evaluate EEG data: **ImageNet EEG** and **THINGS EEG2**. Each pipeline operates independently with its own data assumptions, architecture, and configuration parameters.


##  Repository Layout

```text
synapse-eeg-pipelines/
├── imagenet_codebase/      # ImageNet EEG pipeline (CLIP vocabulary corpus)
│   ├── README.md           # Setup, training, and inference instructions
│   └── ...                 
└── things_codebase/        # THINGS EEG pipeline (ChannelNet + Text‑mapper)
    ├── README.md           # Setup, latent generation, and mapping instructions
    └── ...
```

## Datasets

**ImageNet EEG**
- Uses `.pth` EEG datasets and a CLIP vocabulary corpus.
- The pipeline expects an EEG dataset path (`--dataset`) and a vocabulary file (`--vocab_path`).

**THINGS EEG**
- Uses `.pt` datasets with 4 repeats per image: `eeg`, `label`, `img`, `text`, and optional metadata.
- Includes separate steps for encoder training, latent generation, and text mapping.

## How to use

1. **ImageNet pipeline**
   - Go to `imagenet_codebase/README.md` for training and inference commands.

2. **THINGS pipeline**
   - Go to `things_codebase/README.md` for the full ChannelNet + text‑mapper pipeline.

Each codebase is designed to be runnable independently; use the folder‑specific README as the source of truth for arguments and paths.
