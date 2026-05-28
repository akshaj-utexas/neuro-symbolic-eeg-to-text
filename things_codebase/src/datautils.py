import glob
import os
import re
import torch
import logging
import tqdm
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor
from constants import label_map, id2label


logger = logging.getLogger(__name__)


class EEGDataset:

    # Constructor
    def __init__(
        self, args, fine_tuning=False, eeg_dataset_path=None, dataset_split=None
    ):
        self.args = args
        eeg_dataset_path = eeg_dataset_path or args.eeg_dataset
        dataset_paths = self._resolve_dataset_paths(eeg_dataset_path, dataset_split)
        if len(dataset_paths) == 1:
            loaded = torch.load(dataset_paths[0], weights_only=False)
        else:
            loaded = self._load_and_merge_things(dataset_paths)
        self.fine_tuning = fine_tuning
        self.image_dir = args.image_dir

        if isinstance(loaded, dict) and "dataset" in loaded:
            if len(dataset_paths) > 1:
                raise ValueError(
                    "Multiple legacy datasets are not supported. Provide a single .pth file."
                )
            self.format = "legacy"
            if args.subject != 0:
                self.data = [
                    loaded["dataset"][i]
                    for i in range(len(loaded["dataset"]))
                    if loaded["dataset"][i]["subject"] == args.subject
                ]
            else:
                self.data = loaded["dataset"]
            self.labels = loaded["labels"]
            self.images = loaded["images"]
            sample_eeg = self.data[0]["eeg"]
            if sample_eeg.ndim == 2:
                self.num_timepoints = sample_eeg.shape[0]
                self.num_channels = sample_eeg.shape[1]
            else:
                self.num_timepoints = sample_eeg.shape[-1]
                self.num_channels = sample_eeg.shape[-2]
            self.indices = list(range(len(self.data)))
        elif isinstance(loaded, dict) and all(
            key in loaded for key in ["eeg", "label", "img", "text"]
        ):
            self.format = "things"
            self.eeg = loaded["eeg"]
            self.labels = loaded["label"]
            self.images = loaded["img"]
            self.texts = loaded["text"]
            self.sessions = loaded.get("session")
            self.subjects = loaded.get("subject")
            self.ch_names = loaded.get("ch_names")
            self.times = loaded.get("times")
            self.num_channels = self.eeg.shape[2]
            self.num_timepoints = self.eeg.shape[3]
            self._average_sessions_per_image()
            self.indices = self._build_things_indices()
        else:
            raise ValueError("Unsupported EEG dataset format.")

        # Compute size
        self.size = len(self.indices)

        # Initialize image processor
        self.processor = AutoProcessor.from_pretrained(args.clip_model)

    def _resolve_dataset_paths(self, eeg_dataset_path, dataset_split):
        if eeg_dataset_path is None:
            raise ValueError("An EEG dataset path is required.")
        if isinstance(eeg_dataset_path, (list, tuple)):
            return list(eeg_dataset_path)

        if isinstance(eeg_dataset_path, str) and "," in eeg_dataset_path:
            return [p.strip() for p in eeg_dataset_path.split(",") if p.strip()]

        if isinstance(eeg_dataset_path, str) and os.path.isdir(eeg_dataset_path):
            if dataset_split:
                pattern = os.path.join(
                    eeg_dataset_path, "sub-*", f"{dataset_split}.pt"
                )
                matches = sorted(glob.glob(pattern))
                matches = self._limit_subjects(matches)
                if matches:
                    return matches
                direct = os.path.join(eeg_dataset_path, f"{dataset_split}.pt")
                if os.path.exists(direct):
                    return [direct]

            matches = sorted(glob.glob(os.path.join(eeg_dataset_path, "*.pt")))
            matches = self._limit_subjects(matches)
            if matches:
                return matches
            raise ValueError(
                f"No .pt files found under {eeg_dataset_path} (split={dataset_split})."
            )

        return [eeg_dataset_path]

    def _limit_subjects(self, matches):
        max_subjects = getattr(self.args, "max_subjects", None)
        if not max_subjects or max_subjects <= 0:
            return matches

        def subject_key(path):
            match = re.search(r"sub-(\d+)", path)
            return int(match.group(1)) if match else float("inf")

        sorted_matches = sorted(matches, key=subject_key)
        return sorted_matches[:max_subjects]

    def _load_and_merge_things(self, dataset_paths):
        eeg_list = []
        label_list = []
        img_list = []
        text_list = []
        session_list = []
        subject_list = []
        ch_names = None
        times = None

        for path in dataset_paths:
            loaded = torch.load(path, weights_only=False)
            if not (isinstance(loaded, dict) and all(
                key in loaded for key in ["eeg", "label", "img", "text"]
            )):
                raise ValueError(
                    f"Unsupported dataset format in {path}. Only THINGS EEG can be merged."
                )

            eeg_list.append(loaded["eeg"])
            label_list.append(loaded["label"])
            img_list.append(loaded["img"])
            text_list.append(loaded["text"])

            if "session" in loaded and loaded["session"] is not None:
                session_list.append(loaded["session"])

            subject_match = re.search(r"sub-(\d+)", path)
            if subject_match is not None:
                subject_id = int(subject_match.group(1))
                subject_list.append(
                    np.full((loaded["label"].shape[0],), subject_id, dtype=int)
                )

            if ch_names is None:
                ch_names = loaded.get("ch_names")
            elif loaded.get("ch_names") is not None and loaded.get("ch_names") != ch_names:
                raise ValueError("Channel names do not match across datasets.")

            if times is None:
                times = loaded.get("times")
            elif loaded.get("times") is not None and not np.array_equal(
                loaded.get("times"), times
            ):
                raise ValueError("Time vectors do not match across datasets.")

        merged = {
            "eeg": np.concatenate(eeg_list, axis=0),
            "label": np.concatenate(label_list, axis=0),
            "img": np.concatenate(img_list, axis=0),
            "text": np.concatenate(text_list, axis=0),
            "ch_names": ch_names,
            "times": times,
        }

        if session_list:
            merged["session"] = np.concatenate(session_list, axis=0)
        if subject_list:
            merged["subject"] = np.concatenate(subject_list, axis=0)

        return merged

    def _average_sessions_per_image(self):
        if self.sessions is None:
            return

        groups = {}
        for i in range(len(self.images)):
            image_path = str(self.images[i][0])
            label = int(self.labels[i][0])
            if hasattr(self, "subjects") and self.subjects is not None:
                subject_id = int(self.subjects[i])
                key = (image_path, label, subject_id)
            else:
                key = (image_path, label)
            groups.setdefault(key, []).append(i)

        if len(groups) == len(self.images):
            return

        eeg_list = []
        label_list = []
        img_list = []
        text_list = []
        session_list = []
        subject_list = []
        for key, indices in groups.items():
            eeg_avg = np.mean(self.eeg[indices], axis=0)
            eeg_list.append(eeg_avg)
            label_list.append(self.labels[indices[0]])
            img_list.append(self.images[indices[0]])
            text_list.append(self.texts[indices[0]])
            session_list.append(None)
            if hasattr(self, "subjects") and self.subjects is not None:
                subject_list.append(self.subjects[indices[0]])

        self.eeg = np.stack(eeg_list, axis=0)
        self.labels = np.stack(label_list, axis=0)
        self.images = np.stack(img_list, axis=0)
        self.texts = np.stack(text_list, axis=0)
        self.sessions = np.array(session_list, dtype=object)
        if subject_list:
            self.subjects = np.stack(subject_list, axis=0)
        self.num_channels = self.eeg.shape[2]
        self.num_timepoints = self.eeg.shape[3]

    def _build_things_indices(self):
        class_to_images = {}
        for i in range(len(self.images)):
            class_name = str(self.texts[i][0])
            image_path = str(self.images[i][0])
            class_to_images.setdefault(class_name, set()).add(image_path)

        allowed_images = set()
        for class_name, image_paths in class_to_images.items():
            sorted_images = sorted(image_paths)
            for image_path in sorted_images[:5]:
                allowed_images.add(image_path)

        indices = [
            i for i in range(len(self.images)) if str(self.images[i][0]) in allowed_images
        ]
        return indices

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        index = self.indices[i]
        if self.format == "legacy":
            # Process EEG
            eeg = self.data[index]["eeg"].float().t()
            time_low = max(0, int(self.args.time_low))
            time_high = min(eeg.shape[1], int(self.args.time_high))
            eeg = eeg[:, time_low:time_high]
            eeg = eeg.view(1, eeg.shape[0], time_high - time_low)
            label = self.data[index]["label"]
            image_name = self.images[self.data[index]["image"]]
            image_path = os.path.join(
                self.image_dir, image_name.split("_")[0], image_name + "_sketch.JPEG"
            )
            if self.fine_tuning:
                label_string = label_map[image_name.split("_")[0]]
        else:
            eeg = torch.from_numpy(self.eeg[index]).float()
            if eeg.ndim == 3:
                eeg = eeg.mean(dim=0)
            time_low = max(0, int(self.args.time_low))
            time_high = min(eeg.shape[1], int(self.args.time_high))
            eeg = eeg[:, time_low:time_high]
            eeg = eeg.view(1, eeg.shape[0], time_high - time_low)
            label = int(self.labels[index][0])
            image_rel_path = str(self.images[index][0]).replace(
                "train_images", "training_images"
            )
            image_path = os.path.join(self.image_dir, image_rel_path)
            if self.fine_tuning:
                label_string = str(self.texts[index][0])

        image_raw = Image.open(image_path).convert("RGB")
        image_raw = self.processor(images=image_raw, return_tensors="pt", padding=True)
        image_raw["pixel_values"] = image_raw["pixel_values"].squeeze(0)

        if self.fine_tuning:
            return image_raw, eeg, label_string
        return image_raw, eeg, label


class EncodedEEGDataset(Dataset):
    def __init__(self, base_dataset, indices=None):
        self.base = base_dataset
        self.indices = indices if indices is not None else list(base_dataset.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]

        if self.base.format == "legacy":
            eeg = self.base.data[index]["eeg"].float().t()
            time_low = max(0, int(self.base.args.time_low))
            time_high = min(eeg.shape[1], int(self.base.args.time_high))
            eeg = eeg[:, time_low:time_high]
            eeg = eeg.view(1, eeg.shape[0], time_high - time_low)
            label = self.base.data[index]["label"]
            image_name = self.base.images[self.base.data[index]["image"]]
            caption = label_map.get(image_name.split("_")[0], str(label))
        else:
            eeg = torch.from_numpy(self.base.eeg[index]).float()
            if eeg.ndim == 3:
                eeg = eeg.mean(dim=0)
            time_low = max(0, int(self.base.args.time_low))
            time_high = min(eeg.shape[1], int(self.base.args.time_high))
            eeg = eeg[:, time_low:time_high]
            eeg = eeg.view(1, eeg.shape[0], time_high - time_low)
            label = int(self.base.labels[index][0])
            caption = str(self.base.texts[index][0])

        return eeg, caption, label


class Splitter:

    def __init__(
        self, dataset, split_path, split_num=0, split_name="train", fine_tuning=False
    ):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [
            i
            for i in self.split_idx
            if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600
        ]
        # Compute size
        self.size = len(self.split_idx)
        self.fine_tuning = fine_tuning
        print(f"Total examples in the spllit{split_name} {self.size}")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        if self.fine_tuning:
            img_data, eeg, label_string = self.dataset[self.split_idx[i]]
            return img_data, eeg, label_string
        else:
            img_data, eeg, label = self.dataset[self.split_idx[i]]
            return img_data, eeg, label



        

class EEGFineTuningDataset:

    # Constructor
    def __init__(
        self,
        args,
        tokenizer_path=None,
        max_len=512,
    ):
        
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.max_len = max_len
        if "gemma" in tokenizer_path.lower():
            self.messages = [
                {"role": "user", "content": f"<image> <label_string> Describe this image in one sentence:"},
            ]
            # Gemmas do not have system role
        else:
            self.messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"<image> <label_string> Describe this image in one sentence:"},
            ]
        
        
        # Load EEG signals
        loaded = torch.load(args.eeg_dataset)
        if args.subject != 0:
            self.data = [
                loaded["dataset"][i]
                for i in range(len(loaded["dataset"]))
                if loaded["dataset"][i]["subject"] == args.subject
            ]
        else:
            self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)
        self.image_dir = args.image_dir
        self.id2label = {}

        # Initialize image processor
        self.processor = AutoProcessor.from_pretrained(args.clip_model)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.args.time_low : self.args.time_high, :]
        eeg = eeg.t()
        eeg = eeg.view(1, 128, self.args.time_high - self.args.time_low)
        label = self.data[i]["label"]
        # print(label)
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(
            self.image_dir, image_name.split("_")[0], image_name + "_sketch.JPEG"
        )
        label_string = label_map[image_name.split("_")[0]]
        self.id2label[label] = label_string
        caption_path = os.path.join(
            self.image_dir, image_name.split("_")[0], image_name + "_caption.txt"
        )
        with open(caption_path) as f:
            content = f.readlines()[0].strip()
            content = content.replace("<s>", "")
            content = content.replace("</s>", "")
        message = self.messages+[{"role": "assistant", "content" : content}]
        
        text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        new_text = text.replace("<label_string>", label_string)
        ps = new_text.split("<image>")
        prefix = ps[0]
        suffix = ps[1]
        #print (suffix)
        input_ids1 = self.tokenizer(
            prefix,
            padding="max_length",
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        input_ids2 = self.tokenizer(
            suffix,
            padding="max_length",
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        input_ids1 = input_ids1.squeeze(0)
        input_ids2 = input_ids2.squeeze(0)

        image_raw = Image.open(image_path).convert("RGB")

        image_raw = self.processor(images=image_raw, return_tensors="pt", padding=True)
        image_raw["pixel_values"] = image_raw["pixel_values"].squeeze(0)

        return image_raw, eeg, input_ids1, input_ids2, label_string


class SplitterFineTuning:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # f.dataset = dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [
            i
            for i in self.split_idx
            if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600
        ]
        # Compute size
        self.size = len(self.split_idx)
        print(f"Total examples in the spllit{split_name} {self.size}")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        image_raw, eeg, input_ids1, input_ids2, label_string = self.dataset[self.split_idx[i]]
        return image_raw, eeg, input_ids1, input_ids2, label_string


class Filter:
    # this is to filter datapoints which have valid predicted object labels
    def __init__(self, dataset, eeg_encoder, device = "cpu") -> None:
        dl = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
        self.data = []

        for batch in tqdm.tqdm(dl):
            _, eeg, input_ids1, input_ids2, label_string = batch
            
            eeg = eeg.to(device)
            with torch.no_grad():
                mm_embeds, cls_logits = eeg_encoder(eeg)
            obj_labels = F.softmax(cls_logits, dim=1).argmax(dim=1)
            for i, ls in enumerate(label_string):
                mm_embeds_i = mm_embeds[i]
                input_ids1_i = input_ids1[i]
                input_ids2_i = input_ids2[i]
                label_string_predicted_i = id2label[str(obj_labels[i].item())]
                if label_string_predicted_i == ls:
                    self.data.append([mm_embeds_i, input_ids1_i, input_ids2_i])
        self.size = len(self.data)
        print(f"Total filtered examples {self.size}")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        # Get sample from dataset
        mm_embeds, input_ids1, input_ids2 = self.data[i]
        return mm_embeds, input_ids1, input_ids2
            

class EEGInferenceDataset:

    # Constructor
    def __init__(self, args):
        self.args = args
        # Load EEG signals
        loaded = torch.load(args.eeg_dataset)
        if args.subject != 0:
            self.data = [
                loaded["dataset"][i]
                for i in range(len(loaded["dataset"]))
                if loaded["dataset"][i]["subject"] == args.subject
            ]
        else:
            self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)
        self.image_dir = args.image_dir

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.args.time_low : self.args.time_high, :]
        eeg = eeg.t()
        eeg = eeg.view(1, 128, self.args.time_high - self.args.time_low)
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(
            self.image_dir, image_name.split("_")[0], image_name + ".JPEG"
        )

        # label_strings are only returned as references, not used in predictions
        label_string = label_map[image_name.split("_")[0]]

        caption_path = os.path.join(
            self.image_dir, image_name.split("_")[0], image_name + "_caption.txt"
        )
        # captions are only returned as references, not used in predictions
        with open(caption_path) as f:
            caption_raw = f.readlines()[0].strip()

        return eeg, label_string, caption_raw, image_path


class SplitterInference:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # f.dataset = dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [
            i
            for i in self.split_idx
            if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600
        ]
        # Compute size
        self.size = len(self.split_idx)
        print(f"Total examples in the spllit{split_name} {self.size}")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label_string, expected_caption, image_path = self.dataset[
            self.split_idx[i]
        ]
        return eeg, label_string, expected_caption, image_path
