python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --dataset Serifs \
    --data_root ../serif_dataset \
    --class_weights_path ./class_weights.pth \
    --split overlap \
    --num_steps 80000 \
    --pretrained_dir ./pretrained/ViT-B_16.npz \
    --train_batch_size 12 \
    --eval_batch_size 4 \
    --eval_every 5000 \
    --warmup_steps 2000 \
    --name b16 \
    --model_type ViT-B_16


