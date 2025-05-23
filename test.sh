export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_DISABLE=1

accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port 10410 \
    test.py \
    -opt /users/xiaokangliu/projects/FAR-GRPO/options/test/far/long_video_prediction/FAR_B_Long_dmlab_res64_1000K_bs32.yml