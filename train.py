import torch
import numpy as np
import random
import os
import shutil
import argparse

from trainUtils import train


# define arguments
parser = argparse.ArgumentParser(description="Provide adversarial evaluation pipeline.")
parser.add_argument("--model", help="model to be trained (Resnet18, EfNetB0, ViT_B16)", default='Resnet18', type=str)
parser.add_argument('--pretrained', help="load pretrained weights", action='store_true')
parser.add_argument("--adv_folder", help="adversarial dataset folder", default='../datasets/CIFAR-10-C', type=str)
parser.add_argument('--adv_training', help="include adversarial transform during training", action='store_true')
parser.add_argument("--adv_dataset", help="adversarial dataset", default='gaussian_blur', type=str)
parser.add_argument("--base_lr", help="base learning rate", default=0.01, type=float)
parser.add_argument("--epochs", help="number of training epochs", default=30, type=int)
parser.add_argument("--batch_size", help="training batch size", default=256, type=int)
parser.add_argument("--gamma", help="step LR scheduler gamma", default=0.2, type=float)
parser.add_argument("--step", help="step LR scheduler step size", default=5, type=int)
parser.add_argument("--gpu", help="gpu id", default=0, type=int)
parser.add_argument("--seed", help="random seed", default=50, type=int)
parser.add_argument("--num_workers", help="number of data loading workers", default=8, type=int)

args = parser.parse_args()


# Seed Everything
seed = args.seed

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# set ouput directory
checkpoint_path = './results'
os.makedirs(checkpoint_path, exist_ok=True)


# set training params in args
args.initial_lr = args.base_lr * (args.batch_size/256)
args.name = '{}-lr-{}-g-{}-s-{}-e-{}-adv-{}-pre-{}'.format(args.model, args.base_lr,
                                                            args.gamma, args.step,
                                                            args.epochs, args.adv_training,
                                                            args.pretrained)


# Create Directories
# NOTE: THIS WILL OVERWRITE PREVIOUS RUN OF THE SAME NAME
args.checkpoint_dir = os.path.join(checkpoint_path, args.adv_dataset, args.name)
args.adv_data_path = os.path.join(args.adv_folder, '{}.npy'.format(args.adv_dataset))
args.adv_targets_path = os.path.join(args.adv_folder, 'labels.npy')

if os.path.exists(args.checkpoint_dir) and os.path.isdir(args.checkpoint_dir):
    shutil.rmtree(args.checkpoint_dir)

os. makedirs(args.checkpoint_dir, exist_ok=True)


# train model
log = train(args)