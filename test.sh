# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port 10410 \
    test.py \
    -opt /users/xiaokangliu/projects/FAR-GRPO/options/test/far/long_video_prediction/FAR_M_Long_minecraft_res128_1000K_bs32.yml \
    --resume_from_checkpoint checkpoint-20000