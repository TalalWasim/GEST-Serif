python train.py \
--label 'Serifs' \
--model 'vit_b_32' \
--train_folder '../font_images_train' \
--test_folder '../font_images_test' \
--base_lr 0.01 \
--epochs 30 \
--batch_size 64 \
--gamma 0.2 \
--step 5 \
--gpu 2 \
--seed 50 \
--num_workers 8