"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""
import copy
import json
import os
import glob
import pickle
import base64
import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast, PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tokenizer_image_token

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

END_CUNCK_TOKEN = "<|endofchunk|>"
NUM_IMAGES_TOKEN = 144

class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.offset = 1 if self.tokenizer.encode("\n")[0] == self.tokenizer.bos_token_id else 0
        print(f"The real input id starts at {self.offset}")
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])
            # if DEFAULT_IMAGE_TOKEN in msg and len(msg.split(DEFAULT_IMAGE_TOKEN)) == 2:
            #     msg = DEFAULT_IMAGE_TOKEN + '\n' + msg.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            #     msg = msg.strip()

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()
            else:
                pass
                # raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            # turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
            turn_input_ids, turn_labels = tokenizer_image_token(msg, self.tokenizer)

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_labels))] if (turn_idx % 2) == 0 else list(turn_labels)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids if turn_idx == 0 else turn_input_ids[self.offset:])
            labels.extend(turn_labels if turn_idx == 0 else turn_labels[self.offset:])
        
        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            # labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

            # if "ocr_vqa" not in self.examples[idx]["image"]:
            #     pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
            # else:
            #     pixel_values = None

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

# this dataset class is only targeting on very-large scale SFT dataset like MAmmoth-VL-10M
class FinetuneLargeDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.offset = 1 if self.tokenizer.encode("\n")[0] == self.tokenizer.bos_token_id else 0
        print(f"The real input id starts at {self.offset}")
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        if ".jsonl" in self.instruct_json.suffix:
            self.examples = []
            with open(self.instruct_json, 'r') as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
            
        elif ".json" in self.instruct_json.suffix:
            with open(self.instruct_json, "r") as f:
                self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        json_path = os.path.join(os.path.join(os.path.dirname(self.instruct_json), "instruction_si_split"), self.examples[idx]["path"])
        example = json.load(open(json_path))
        conversation = example["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])
            # if DEFAULT_IMAGE_TOKEN in msg and len(msg.split(DEFAULT_IMAGE_TOKEN)) == 2:
            #     msg = DEFAULT_IMAGE_TOKEN + '\n' + msg.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            #     msg = msg.strip()

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()
            else:
                pass
                # raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            # turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
            turn_input_ids, turn_labels = tokenizer_image_token(msg, self.tokenizer)

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_labels))] if (turn_idx % 2) == 0 else list(turn_labels)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids if turn_idx == 0 else turn_input_ids[self.offset:])
            labels.extend(turn_labels if turn_idx == 0 else turn_labels[self.offset:])
        
        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in example:
            image_path = Path(example["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            # labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            try:
                pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
            except Exception:
                pixel_values = None

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = ("jpg" in example or "jpeg" in example or "png" in example["path"])
            n_words = example["length"]
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class PreTrainDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        pretrain_pkl: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        train_num_samples: int,
    ) -> None:
        super().__init__()
        self.pretrain_pkl = pretrain_pkl
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "pretrain"

        # Load paths for all pickle files
        self.examples = []
        for subset_data_path in self.pretrain_pkl.split(":"):
            if "pkl" in subset_data_path:
                subset_examples = glob.glob(os.path.join(subset_data_path, '**', '*.pkl'), recursive=True)
            elif "json" in subset_data_path:
                subset_examples = glob.glob(os.path.join(subset_data_path, '**', '*packed.json'), recursive=True)
            print("{} contains {} examples".format(subset_data_path, len(subset_examples)))
            
            self.examples += subset_examples
            print("Now the dataset is expanded to contain {} examples".format(len(self.examples)))

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        if self.examples[idx].endswith(".pkl"):
            with open(self.examples[idx], 'rb') as f:
                data_pkl = pickle.load(f)
        elif self.examples[idx].endswith("packed.json"):
            with open(self.examples[idx], 'r') as f:
                data_json = json.load(f)
            data_samples = [self.preprocess_obelics_interleaved(sample) for sample in data_json["data"]]
            data_samples = list(filter(None, data_samples))
            data_pkl = self.concat_document(data_samples, pad_token_id=self.tokenizer.pad_token_id)
        image_tensors = [self.image_transform(image) for image in data_pkl['image_tensors']]
        data_pkl['image_tensors'] = image_tensors
        labels = copy.deepcopy(data_pkl['input_ids'])
        labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        data_pkl['labels'] = labels
        
        return data_pkl

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
    

    def preprocess_obelics_interleaved(
        self,
        info,
    ):
        """
        Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
        is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
        """
        # image transform includes the padding process and normalization process
        
        texts = info["texts"]

        # load images first to find which ones are valid
        image_tensors, idxs = [], []
        for i, sample_image in enumerate(info["image_info"]):
            if not sample_image or "image_base64" not in sample_image:
                idxs.append(i)
                continue
            image_base64 = sample_image["image_base64"]
            rawbytes = base64.b64decode(image_base64)

            # filter to images >= 10KB
            # if len(rawbytes) // 1000 <= MIN_KB:
            #     continue

            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            if image.size[0] == 1 or image.size[1] == 1:
                continue
            
            # no need to resize

            # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # image_tensors.append(image_tensor)
            image_tensors.append(image)
            idxs.append(i)

        if len(image_tensors) == 0 or len(image_tensors) >= 10:
            return None

        # preprocess and pad images
        texts = [texts[idx] for idx in idxs]
        if len(image_tensors) >= 2:
            texts =  [DEFAULT_IMAGE_TOKEN+"\n" if x is None else x.replace(DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER)+END_CUNCK_TOKEN for x in texts]
        else:
            texts =  [DEFAULT_IMAGE_TOKEN+"\n" if x is None else x.replace(DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER) for x in texts]
        corpus = "".join(texts)

        input_ids, labels = tokenizer_image_token(corpus, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        if input_ids[0] != self.tokenizer.bos_token_id:
            input_ids = torch.cat([torch.LongTensor([self.tokenizer.bos_token_id]), input_ids])
        
        # if input_ids.shape[-1] > 6000:
        #     return None

        assert len(image_tensors) == ((input_ids == IMAGE_TOKEN_INDEX).sum())

        # Some examples contain many images at the end, which will not be attended on due to auto-regressive learning, truncate them
        # if input_ids[-1] == IMAGE_TOKEN_INDEX:
        #     input_ids = truncate_end_image_token(input_ids)
        
        # recompute the num of images
        final_num_images = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
        
        # if all the images are at the end, then the doc becomes text-only doc, discard it
        if final_num_images == 0:
            return None
        
        assert len(image_tensors[:final_num_images]) == final_num_images
        
        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.tokenizer.eos_token_id])])
        
        # assert bos token
        assert input_ids[0] == self.tokenizer.bos_token_id

        length = (NUM_IMAGES_TOKEN - 1) * len(image_tensors) + input_ids.shape[-1]

        return (image_tensors[:final_num_images], input_ids, length)

    def concat_document(self, documents, pad_token_id, context_len=4096):
        new_input_ids = []
        # new_labels = []
        new_image_tensors = []
        new_length = []
        for (image_tensors, input_ids, length) in documents:
            new_image_tensors += image_tensors
            new_input_ids.append(input_ids)
            # new_labels.append(labels)
            new_length.append(length)

        new_input_ids = torch.cat(new_input_ids)
        # new_labels = copy.deepcopy(new_input_ids)
        # new_labels[new_labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX

        if len(new_image_tensors) == 0:
            return None
        
        assert (new_input_ids == IMAGE_TOKEN_INDEX).sum() == len(new_image_tensors)
        assert (sum(new_length) - new_input_ids.shape[-1]) == (NUM_IMAGES_TOKEN - 1) * len(new_image_tensors)

        if new_input_ids.shape[-1] < context_len:
            new_input_ids = torch.cat([new_input_ids, torch.full((context_len - new_input_ids.shape[-1],), pad_token_id, dtype=new_input_ids.dtype)])
            # new_labels = torch.cat([new_labels, torch.full((context_len - new_labels.shape[-1],), IGNORE_INDEX, dtype=new_labels.dtype)])

        # attention_mask = torch.zeros((context_len, context_len), dtype=torch.bool)
        # start = 0
        # for length in new_length:
        #     end = start + length
        #     attention_mask[start:end, start:end] = True
        #     start += length

        # return {"image_tensors": new_image_tensors, "input_ids": new_input_ids, "labels": new_labels, "attention_mask": attention_mask, "lengths": new_length}
        return {"image_tensors": new_image_tensors, "input_ids": new_input_ids, "lengths": new_length}