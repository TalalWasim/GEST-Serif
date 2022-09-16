#!/bin/bash
#SBATCH --job-name=resnet_gaussian
#SBATCH --partition=default-short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=14900
#SBATCH --gres=gpu:1

python train.py \
--model 'ViTB16' \
--adv_folder '../datasets/CIFAR-10-C' \
--adv_dataset 'gaussian_blur' \
--base_lr 0.01 \
--epochs 40 \
--batch_size 256 \
--gamma 0.2 \
--step 5 \
--gpu 0 \
--seed 50