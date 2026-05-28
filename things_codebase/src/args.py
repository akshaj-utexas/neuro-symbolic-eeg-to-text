import argparse


def get_args_for_encoder_training():
    # Define options
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    ### BLOCK DESIGN ###
    # Data
    parser.add_argument(
        "--eeg_dataset", default=None, help="EEG dataset path"
    )  # 5-95Hz
    parser.add_argument(
        "--eeg_train_dataset",
        default=None,
        help="EEG train dataset path (overrides --eeg_dataset)",
    )
    parser.add_argument(
        "--eeg_val_dataset",
        default=None,
        help="EEG validation dataset path",
    )
    parser.add_argument(
        "--eeg_test_dataset",
        default=None,
        help="EEG test dataset path",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit to first N subjects when loading from a directory.",
    )
    parser.add_argument("--image_dir", default=None, help="ImageNet dataset path")
    # Splits
    parser.add_argument(
        "--splits_path", default=None, help="splits path"
    )  # All subjects
    ### BLOCK DESIGN ###
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the model checkpoints and logs.",
    )

    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    parser.add_argument(
        "-sn", "--split_num", default=0, type=int, help="split number"
    )  # leave this always to zero.

    # Subject selecting
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )
    # Training options
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda")

    # train args

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "--save_steps",
        default=5000,
        type=int,
        help="Number of steps between saving checkpoints.",
    )
    parser.add_argument(
        "--logging_steps", default=30, type=int, help="Number of steps between logging."
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        type=str,
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--weight_decay", default=0.001, type=float, help="Weight decay to apply."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=0.3,
        type=float,
        help="Max gradient norm to clip gradients.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.3,
        type=float,
        help="Ratio of total steps to perform linear learning rate warmup.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Whether to group samples of roughly the same length together.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        type=str,
        help="Type of learning rate scheduler.",
    )
    # Parse arguments
    args = parser.parse_args()
    return args


def get_args_for_text_mapper_training():
    parser = argparse.ArgumentParser(
        description="Train EEG-to-CLIP-text mapper with multi-hot supervision."
    )

    parser.add_argument(
        "--encoded_eeg_dataset",
        type=str,
        required=True,
        help="Path to ChannelNet-encoded EEG embeddings (.pth/.pt).",
    )
    parser.add_argument(
        "--vocab_corpus",
        type=str,
        required=True,
        help="Path to CLIP text corpus file produced by build_corpus.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save mapper checkpoints.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha weighting for positives.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma focusing parameter.",
    )
    parser.add_argument(
        "--use_scaling",
        action="store_true",
        help="Apply learnable temperature scaling to cosine logits.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to reserve for validation (0 disables).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of training samples.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def get_args_for_latent_encoding():
    parser = argparse.ArgumentParser(
        description="Encode EEG samples into CLIP-aligned latents using ChannelNet."
    )

    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to the trained ChannelNet encoder checkpoint directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the encoded EEG dataset (.pth).",
    )
    parser.add_argument(
        "--eeg_dataset", default=None, help="EEG dataset path"
    )
    parser.add_argument(
        "--eeg_train_dataset",
        default=None,
        help="EEG train dataset path (overrides --eeg_dataset)",
    )
    parser.add_argument(
        "--splits_path", default=None, help="splits path"
    )
    parser.add_argument(
        "-sn", "--split_num", default=0, type=int, help="split number"
    )
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit to first N subjects when loading from a directory.",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Image directory (required by dataset loader but not used for encoding).",
    )
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args


def get_args_for_text_mapper_inference():
    parser = argparse.ArgumentParser(
        description="Run ChannelNet + text mapper inference on test data."
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to a merged test .pt file containing eeg/img/text (preferred).",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default=None,
        help="Root directory containing sub-XX/test.pt files.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Subject IDs to process (e.g., 1 2 3 4 5).",
    )
    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to the trained ChannelNet encoder checkpoint directory.",
    )
    parser.add_argument(
        "--text_mapper_ckpt",
        type=str,
        required=True,
        help="Path to the trained text mapper checkpoint (text_mapper_best.pt).",
    )
    parser.add_argument(
        "--vocab_corpus",
        type=str,
        default=None,
        help="Path to CLIP text corpus file produced by build_corpus.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/text-mapper-inference",
        help="Directory to save per-subject outputs when using --test_root.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for merged test output (defaults to overwriting --test_path).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Root Image_set directory containing captions (required for --test_root).",
    )
    parser.add_argument(
        "--use_precomputed_latents",
        action="store_true",
        help="Use eeg_clip_latent from --test_path (skip encoder).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--top_k", type=int, default=15, help="Top-K words to save per sample."
    )
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )

    args = parser.parse_args()
    return args
