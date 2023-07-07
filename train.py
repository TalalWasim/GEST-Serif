import torch
import numpy as np
import random
import os
import shutil
import argparse

from trainUtils import train


# define arguments
parser = argparse.ArgumentParser(description="Provide GEST Fonts/Serifs training pipeline.")
parser.add_argument("--label", help="training label (fonts/serifs)", default='serifs', type=str)
parser.add_argument("--model", help="model to be trained", default='resnet18', type=str)
parser.add_argument("--train_folder", help="training dataset folder", default='../font_images_train', type=str)
parser.add_argument('--test_folder', help="testing dataset folder", default='../font_images_test', type=str)

# optimizer
parser.add_argument("--base_lr", help="base learning rate", default=0.001, type=float)
parser.add_argument("--epochs", help="number of training epochs", default=30, type=int)
parser.add_argument("--batch_size", help="training batch size", default=256, type=int)
parser.add_argument("--gamma", help="step LR scheduler gamma", default=0.1, type=float)
parser.add_argument("--step", help="step LR scheduler step size", default=5, type=int)

# transforms
parser.add_argument("--resize_size", help="size to which image is resized", default=600, type=int)
parser.add_argument("--crop_size", help="size to which image is cropped", default=448, type=int)

# misc
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
args.name = '{}-{}-lr-{}-g-{}-s-{}-e-{}'.format(args.label, args.model,
                                                args.base_lr, args.gamma,
                                                args.step, args.epochs)


# Create Directories
# NOTE: THIS WILL OVERWRITE PREVIOUS RUN OF THE SAME NAME
args.checkpoint_dir = os.path.join(checkpoint_path, args.name)

if os.path.exists(args.checkpoint_dir) and os.path.isdir(args.checkpoint_dir):
    shutil.rmtree(args.checkpoint_dir)

os. makedirs(args.checkpoint_dir, exist_ok=True)


# train model
log = train(args)