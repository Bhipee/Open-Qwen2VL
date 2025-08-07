CKPTID=$1
STAGE=$2
BSZ=$3
PER_GPU_BSZ=$4

torchrun --nproc_per_node 8 scripts/pretrain.py \
  --stage ${STAGE} \
  --model.type "one-stage+7b" \
  --model.model_id qwen3-30b-a3b-continue-training-${CKPTID} \
  --model.arch_specifier "no-align+avgpool" \
  --model.vision_backbone_id "siglip-vit-so400m-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "qwen3-30b-a3b-megablock" \
  --model.pretrain_global_batch_size ${BSZ} \
  --model.pretrain_per_device_batch_size ${PER_GPU_BSZ} \
  --model.pretrain_epochs 1 \
  --mount_path /lustre/projects/polyullm/models/Qwen3 \
  --run_root_dir checkpoints/ \
  --dataset.type "pretrain" \
  --dataset.dataset_root_dir /home/projects/polyullm/guanghao/Open-Qwen2VL-main/Open-Qwen2VL-Data/datacomp_hq_single_pkl_pil
  # --dataset.dataset_root_dir /home/projects/polyullm/guanghao/Open-Qwen2VL-main/Open-Qwen2VL-Data/datacomp_hq_single_pkl_pil:/home/projects/polyullm/guanghao/Open-Qwen2VL-main/Open-Qwen2VL-Data/ccs_single_pkl_pil
  # --mount_path  /lustre/projects/polyullm/models/Qwen3 \
