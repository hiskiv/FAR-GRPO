#!/bin/bash

# Destination directory
DEST="/storage/xiaokangliu/project_backup/FAR_GRPO_exp"

# List of directories to move
DIRS=(
  "./v4_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
  "./v5_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
  "./v6_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
  "./v7_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
  "./v8_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
  "./v9_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
  "./v11_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
  "./lora+kl-v8-fullmse-56fr-20steps_FAR_M_Long_minecraft_res128_1000K_bs32_GRPO"
)

# Move each directory
for dir in "${DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "Moving $dir to $DEST"
    mv "$dir" "$DEST"
  else
    echo "Warning: $dir does not exist or is not a directory."
  fi
done

