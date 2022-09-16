python train.py \
--model 'Resnet18' \
--adv_folder '../datasets/CIFAR-10-C' \
--adv_dataset 'gaussian_blur' \
--base_lr 0.01 \
--epochs 3 \
--batch_size 512 \
--gamma 0.2 \
--step 5 \
--gpu 0 \
--seed 50