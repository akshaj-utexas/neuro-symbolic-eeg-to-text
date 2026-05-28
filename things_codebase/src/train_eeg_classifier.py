import os
import random
import json
import torch
import numpy as np
from tqdm import tqdm
from datautils import EEGDataset, Splitter
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from args import get_args_for_encoder_training
from loss import MSELoss
from transformers import (
    Trainer,
    TrainingArguments,
    AutoProcessor,
    CLIPVisionModelWithProjection,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split, Subset


def set_seed(seed):
    """Set seed for reproducibility"""
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable to ensure reproducibility


class EEGEncoderTrainer(Trainer):
    def __init__(
        self,
        emb_loss_fn=None,
        cls_loss_fn=None,
        clip_model=None,
        data_loaders=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_loss_fn = emb_loss_fn
        self.clip_model = clip_model
        self.data_loaders = data_loaders
        self.device = "cpu"

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.model.train()
        img_data, eeg, _labels = inputs
        image_embeddings = self.clip_model(
            pixel_values=img_data["pixel_values"]
        ).image_embeds
        emb_output, _ = model(eeg)
        emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
        loss = emb_loss
        self.device = eeg.device
        return (loss, emb_output) if return_outputs else loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.data_loaders["test"]

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset=None)
        eval_loss = 0
        for batch in tqdm(eval_dataloader):
            image_raw, eeg_data, labels = batch
            image_raw = image_raw.to(self.device)
            eeg_data = eeg_data.to(self.device)
            labels = labels.to(self.device)
            image_embeddings = self.clip_model(
                pixel_values=image_raw["pixel_values"]
            ).image_embeds
            emb_output, _ = self.model(eeg_data)
            emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
            loss = emb_loss
            eval_loss += loss.item()
        print({"eval_loss": eval_loss})

        # Do testing
        test_dataloader = self.get_test_dataloader(test_dataset=None)
        test_loss = 0
        for batch in tqdm(test_dataloader):
            image_raw, eeg_data, labels = batch
            image_raw = image_raw.to(self.device)
            eeg_data = eeg_data.to(self.device)
            labels = labels.to(self.device)
            image_embeddings = self.clip_model(
                pixel_values=image_raw["pixel_values"]
            ).image_embeds
            emb_output, _ = self.model(eeg_data)
            emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
            loss = emb_loss
            test_loss += loss.item()
        print({"test_loss": test_loss})

        return {"eval_loss": -eval_loss}


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def _print_dataset_summary(name, dataset):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    total = len(dataset)
    print(f"{name} samples: {total}")
    if getattr(base, "format", None) == "things":
        try:
            unique_images = len(np.unique(base.images))
        except Exception:
            unique_images = "unknown"
        try:
            labels = base.labels
            if isinstance(labels, torch.Tensor):
                unique_labels = int(torch.unique(labels).numel())
            else:
                unique_labels = int(np.unique(labels).shape[0])
        except Exception:
            unique_labels = "unknown"
        session_info = "present" if base.sessions is not None else "missing"
        subject_info = "missing"
        if getattr(base, "subjects", None) is not None:
            try:
                subject_info = int(np.unique(base.subjects).shape[0])
            except Exception:
                subject_info = "unknown"
        print(
            f"{name} unique images: {unique_images}, unique labels: {unique_labels}, sessions: {session_info}, subjects: {subject_info}"
        )


def main():
    args = get_args_for_encoder_training()
    set_seed(42)
    # processor = AutoProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model)
    clip_model.to(args.device)
    clip_model.requires_grad_(False)
    set_gradients(clip_model, False)
    clip_model.eval()

    if args.eeg_train_dataset or args.eeg_test_dataset:
        train_dataset = EEGDataset(
            args=args,
            eeg_dataset_path=args.eeg_train_dataset or args.eeg_dataset,
            dataset_split="train",
        )
        if args.eeg_val_dataset:
            val_dataset = EEGDataset(
                args=args, eeg_dataset_path=args.eeg_val_dataset, dataset_split="val"
            )
        else:
            val_size = max(1, int(0.1 * len(train_dataset)))
            train_size = len(train_dataset) - val_size
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size], generator=generator
            )

        if args.eeg_test_dataset:
            test_dataset = EEGDataset(
                args=args, eeg_dataset_path=args.eeg_test_dataset, dataset_split="test"
            )
        else:
            test_dataset = val_dataset

        _print_dataset_summary("train", train_dataset)
        _print_dataset_summary("val", val_dataset)
        _print_dataset_summary("test", test_dataset)

        loaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
            ),
            "test": DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
            ),
        }
        dataset = train_dataset
    else:
        dataset = EEGDataset(args=args)
        loaders = {
            split: DataLoader(
                Splitter(
                    dataset,
                    split_path=args.splits_path,
                    split_num=args.split_num,
                    split_name=split,
                ),
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
            )
            for split in ["train", "val", "test"]
        }
        _print_dataset_summary("full", dataset)

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    input_height = getattr(base_dataset, "num_channels", 128)
    time_low = max(0, int(args.time_low))
    time_high = min(getattr(base_dataset, "num_timepoints", args.time_high), int(args.time_high))
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

    config = EEGModelConfig(
        input_height=input_height, input_width=input_width, num_classes=num_classes
    )

    config.save_pretrained(args.output)
    model = ChannelNetModel(config=config)

    training_args_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        load_best_model_at_end=True,
        save_strategy="epoch",
        eval_strategy="epoch",
    )
    try:
        training_args_kwargs["group_by_length"] = args.group_by_length
        training_arguments = TrainingArguments(**training_args_kwargs)
    except TypeError as exc:
        if "group_by_length" not in str(exc):
            raise
        training_args_kwargs.pop("group_by_length", None)
        try:
            training_arguments = TrainingArguments(**training_args_kwargs)
        except ValueError as inner_exc:
            if "OptimizerNames" not in str(inner_exc):
                raise
            if training_args_kwargs.get("optim") == "adamw_hf":
                training_args_kwargs["optim"] = "adamw_torch"
                training_arguments = TrainingArguments(**training_args_kwargs)
            else:
                raise
    trainer = EEGEncoderTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=dataset,
        emb_loss_fn=MSELoss(),
        cls_loss_fn=torch.nn.CrossEntropyLoss(),
        data_loaders=loaders,
        clip_model=clip_model,
    )
    trainer.train()
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
