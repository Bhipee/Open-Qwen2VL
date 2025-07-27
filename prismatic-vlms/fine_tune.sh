CKPT_PATH=$1
CKPTID=$2
DATAPATH=$3

# mount_path: you can change it to the path where you pre-download the pre-trained LLMs, if you will to download the model directly from HF hub, please use the holder name of the model, i.e. Qwen for Qwen-series models. Then the mount_path will be concatenated with model_id

# trackers: you can change to ["jsonl",] ["wandb",] ["jsonl","wandb"] to decide whether to visualize your training on wandb


# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --stage "finetune" \
  --model.type "one-stage+7b" \
  --model.model_id qwen2.5-1.5b-instruct-continue-training-${CKPTID} \
  --model.arch_specifier full-align+729-avgpool \
  --model.vision_backbone_id "siglip-vit-so400m-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id qwen2.5-1.5b-instruct \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 2 \
  --mount_path Qwen \
  --run_root_dir checkpoints \
  --dataset.type "llava-v15" \
  --pretrained_checkpoint ${CKPT_PATH}/checkpoints/latest-checkpoint.pt \
  --dataset.finetune_stage_components=["${DATAPATH}","llava/images/"]