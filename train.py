import torch
import numpy as np
import random
import os
import shutil
import argparse

from trainUtils import load_dataset, get_model, get_optimizer_scheduler, train

# define arguments
parser = argparse.ArgumentParser(description="Provide adversarial evaluation pipeline.")
parser.add_argument("--model", help="model to be trained (Resnet18, EfNetB0, ViT_B16)", default='Resnet18', type=str)
parser.add_argument('--pretrained', help="load pretrained weights", action='store_true')
parser.add_argument('--adv_training', help="include adversarial transform during training", action='store_true')
parser.add_argument("--adv_dataset", help="adversarial dataset", default='gaussian_blur', type=str)
parser.add_argument("--lr", help="base learning rate", default=0.01, type=float)
parser.add_argument("--epochs", help="number of training epochs", default=30, type=int)
parser.add_argument("--batchsize", help="training batch size", default=256, type=int)
parser.add_argument("--gamma", help="step LR scheduler gamma", default=0.2, type=float)
parser.add_argument("--step", help="step LR scheduler step size", default=5, type=int)
parser.add_argument("--gpu", help="gpu id", default=0, type=int)
parser.add_argument("--seed", help="random seed", default=50, type=int)

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


# set device
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")


# set ouput directory
checkpoint_path = './results'
os.makedirs(checkpoint_path, exist_ok=True)


# set training params
model_name = args.model
adv_folder = '../datasets/CIFAR-10-C'
adv_dataset = args.adv_dataset
adv_training = args.adv_training

num_epochs = args.epochs
base_lr = args.lr
train_batch_size = args.batchsize
initial_lr = base_lr * (train_batch_size/256)
gamma = args.gamma # StepLR gamma
step = args.step # StepLR step size
pretrained = args.pretrained

name = '{}-LR-{}-gamma-{}-step-{}-epochs-{}-advdata-{}-advtrain-{}'.format(model_name, initial_lr, gamma, step,
                                                                            num_epochs, adv_dataset, adv_training)



# Create Directories
# NOTE: THIS WILL OVERWRITE PREVIOUS RUN OF THE SAME NAME
checkpoint_dir = os.path.join(checkpoint_path, name)
adv_data_path = os.path.join(adv_folder, '{}.npy'.format(adv_dataset))
adv_targets_path = os.path.join(adv_folder, 'labels.npy')

if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

os. makedirs(checkpoint_dir, exist_ok=True)


# Build dataloaders
train_loader, test_loader, adv_loaders, CLASSES = load_dataset(train_batch_size, adv_training,
                                                                adv_dataset, adv_data_path, adv_targets_path)


# get model, optimizer and scheduler
model = get_model(model_name, len(CLASSES), pretrained=False)
optimizer, scheduler = get_optimizer_scheduler(model, initial_lr, step, gamma)


# train model
log = train(1, num_epochs, name, checkpoint_dir, model, train_loader, test_loader, adv_loaders, optimizer, scheduler)