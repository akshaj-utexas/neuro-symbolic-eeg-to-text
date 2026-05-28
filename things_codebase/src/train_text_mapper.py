import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import nltk

from args import get_args_for_text_mapper_training
from trainer import Stage1_5Dataset
from models import SimilarityRefiner, FocalLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_nltk_resources():
    for res in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
        nltk.download(res, quiet=True)


def _split_dataset(dataset, val_split, seed):
    if val_split <= 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def main():
    args = get_args_for_text_mapper_training()
    set_seed(args.seed)
    _ensure_nltk_resources()

    os.makedirs(args.output, exist_ok=True)

    dataset = Stage1_5Dataset(args.encoded_eeg_dataset, args.vocab_corpus)
    if args.max_samples:
        max_samples = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))

    train_dataset, val_dataset = _split_dataset(dataset, args.val_split, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    vocab_info = torch.load(args.vocab_corpus, map_location="cpu")
    model = SimilarityRefiner(
        vocab_info["embeddings"],
        input_dim=512,
        hidden_dim=args.hidden_dim,
        use_scaling=args.use_scaling,
    )

    model.to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    best_val_loss = None
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for eeg, target in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            eeg = eeg.to(args.device)
            target = target.to(args.device)
            optimizer.zero_grad()
            logits, _ = model(eeg)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        print({"epoch": epoch + 1, "train_loss": avg_train_loss})

        if val_loader is None:
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg, target in val_loader:
                eeg = eeg.to(args.device)
                target = target.to(args.device)
                logits, _ = model(eeg)
                loss = criterion(logits, target)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        print({"epoch": epoch + 1, "val_loss": avg_val_loss})

        if best_val_loss is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "use_scaling": args.use_scaling,
                    "vocab_corpus": args.vocab_corpus,
                },
                os.path.join(args.output, "text_mapper_best.pt"),
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "use_scaling": args.use_scaling,
            "vocab_corpus": args.vocab_corpus,
        },
        os.path.join(args.output, "text_mapper_last.pt"),
    )


if __name__ == "__main__":
    main()
