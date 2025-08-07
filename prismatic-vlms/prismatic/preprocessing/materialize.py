"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.conf import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.preprocessing.datasets import AlignDataset, FinetuneDataset, PreTrainDataset, FinetuneLargeDataset
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling, PaddedCollatorForMMLanguageModeling, SharedEpoch, DataInfo, detshuffle2, ResampledShards2
from prismatic.util.data_utils import get_dataset_size, count_samples, log_and_continue, group_by_keys_nothrow, tarfile_to_samples_nothrow, pytorch_worker_seed, world_info_from_env

import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "moe-finetune": FinetuneDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset, "large-finetune": FinetuneLargeDataset, "moe-pretrain": PreTrainDataset, "pretrain": PreTrainDataset, "full-pretrain": PreTrainDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage in ["moe-finetune", "finetune", "large-finetune", "full-finetune"]:
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage in ["moe-pretrain", "pretrain", "full-pretrain"]:
        collator = PaddedCollatorForMMLanguageModeling(
            tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
        )
        dataset = dataset_cls(
            dataset_root_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            train_num_samples=dataset_cfg.train_num_samples,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")


def get_wds_datainfo(
    stage: str,
    gloabl_batch_size: int,
    per_device_batch_size: int,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    
    dataset_root_dir = dataset_cfg.dataset_root_dir
    train_wds_datainfo = get_mmc4_dataset(dataset_cfg, gloabl_batch_size, per_device_batch_size, image_transform, tokenizer)

    return train_wds_datainfo


def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def preprocess_obliecs_interleaved(
    sample,
    tokenizer,
    image_transform,
    min_num_images,
    max_num_images,
    max_tokens=4096,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    # image transform includes the padding process and normalization process
    info = json.loads(sample[0])

    image_urls = info["images"]
    texts = info["texts"]

    # load images first to find which ones are valid
    valid_images, valid_image_indices = [], []
    for i, sample_image in enumerate(info["image_info"]):
        if not sample_image or "image_base64" not in sample_image:
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        # filter to images >= 10KB
        # if len(rawbytes) // 1000 <= MIN_KB:
        #     continue

        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        valid_images.append(image)
        valid_image_indices.append(i)

    # if len(valid_image_indices) == 0:
    #     raise ValueError("No images in sample")

    # sim_matrix = np.array(sim_matrix)  # of shape images x sentences
    # sim_matrix = sim_matrix[valid_image_indices]

    # # negate the similarities to turn then into costs
    # cost_matrix = -sim_matrix
    # # find one to one assignements
    # image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

    # images, sentence_ixs = [], []
    # for i, sim_ix in zip(image_indices, sentence_indices):
    #     sim_score = sim_matrix[i][sim_ix]

    #     if sim_score < sim_threshold:
    #         continue

    #     images.append(valid_images[i])
    #     sentence_ixs.append(sim_ix)

    if len(valid_images) == 0:
        raise ValueError("No images in sample")

    # return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    # preprocess and pad images
    pixel_values = [image_transform(i) for i in valid_images]

    # sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    # if len(images_tensors) < max_num_images:
    #     zero_padding = torch.zeros(
    #         (
    #             max_num_images - len(images_tensors),
    #             N_CHANNELS,
    #             images_tensors[0].shape[1],
    #             images_tensors[0].shape[2],
    #         ),
    #         dtype=torch.float,
    #     )
    #     images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens

    # Weizhi: We do not need to insert the eot token
    # for ix in sentence_ixs:
    #     sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    texts =  ["<image>\n" if x is None else x.replace("<image>", "image")+"<|end|>" for x in texts]
    corpus = "".join(texts)
    input_ids, labels = tokenizer_image_token(corpus, tokenizer)

    # add bos token
    if input_ids[0] != tokenizer.bos_token_id:
        input_ids.insert(0, tokenizer.bos_token_id)
        labels.insert(0, tokenizer.bos_token_id)

    # add eos token
    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    # padding
    current_len = len(input_ids)
    assert tokenizer.pad_token_id is not None
    input_ids += [tokenizer.pad_token_id] * (max_tokens - current_len)
    labels += [IGNORE_INDEX] * (max_tokens - current_len)
    input_ids = input_ids[:max_tokens]
    labels = labels[:max_tokens]
    input_ids, labels = torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return (pixel_values, input_ids, labels, attention_mask)

    # text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    # text = (
    #     text.replace(" <|endofchunk|>", "<|endofchunk|>")
    #     .replace("<image> ", "<image>")
    #     .replace(" <image>", "<image>")
    # )
    # text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    # tokenizer.padding_side = "right"
    # text_tensor = tokenizer(
    #     corpus,
    #     max_length=max_tokens,
    #     truncation=True,
    #     padding="max_length",
    #     return_tensors="pt",
    # )

    # reject sequences with too few images (after truncation)
    # num_images = torch.count_nonzero(
    #     text_tensor["input_ids"]
    #     == tokenizer.additional_special_tokens_ids[
    #         tokenizer.additional_special_tokens.index("<image>")
    #     ]
    # )
    # if num_images < min_num_images:
    #     raise ValueError(f"Fewer than {min_num_images} images in sample")
    # elif (
    #     num_images == 1 and random.random() <= 0.5
    # ):  # 50% chance of keeping single image samples
    #     raise ValueError("Only one image in sample")

    # # avoid the situation where there's one <image> token and it's at the end
    # if (
    #     num_images == 1
    #     and text_tensor["input_ids"][:, -1]
    #     == tokenizer.additional_special_tokens_ids[
    #         tokenizer.additional_special_tokens.index("<image>")
    #     ]
    # ):
    #     raise ValueError(
    #         "Only one image at the end of sample, so labels will all be -100"
    #     )

    # return (
    #     images_tensors,
    #     (input_ids, attention_mask, labels),
    # )


def get_mmc4_dataset(dataset_cfg, gloabl_batch_size, per_device_batch_size, image_transform, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for MMC4 / ChatGPT sequences
    """
    input_shards = dataset_cfg.dataset_root_dir
    assert input_shards is not None
    resampled = getattr(dataset_cfg, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    
    num_samples = None
    if not num_samples:
        num_samples = dataset_cfg.train_num_samples
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = functools.partial(
        preprocess_obliecs_interleaved,
        tokenizer=tokenizer,
        image_transform=image_transform,
        min_num_images=dataset_cfg.min_num_images,
        max_num_images=dataset_cfg.max_num_images,
    )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=dataset_cfg.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(per_device_batch_size, partial=False),
        ]
    )
    local_rank, global_rank, world_size = world_info_from_env()
    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= dataset_cfg.workers * world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = per_device_batch_size * world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, dataset_cfg.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=dataset_cfg.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    print(dataloader.num_batches, dataloader.num_samples)

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    labels = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
        labels.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    
    for x in insert_separator(prompt_chunks, [IGNORE_INDEX] * (offset + 1)):
        labels.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids, labels


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result