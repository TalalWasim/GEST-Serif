from operator import ne
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import random
import os
import shutil
import pickle
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, cm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




#############################################
#### Single Epoch Training Function
#############################################

def train_for_epoch(model, train_loader, criterion, optimizer, scheduler, device):

    # put model in train mode
    model.train()

    # keep track of the training losses during the epoch
    train_losses = []
    train_predictions = []
    train_true = []

    # Training
    for batch, targets in train_loader:

        # Move the training data to the GPU
        batch = batch.to(device, dtype=torch.float)
        train_true.extend(targets.numpy())
        targets = targets.to(device)

        # clear previous gradient computation
        optimizer.zero_grad()

        # forward propagation
        predictions = model(batch)

        # calculate the loss
        loss = criterion(predictions, targets) # calculating loss

        # backpropagate to compute gradients
        loss.backward()

        # update model weights
        optimizer.step()

        # update running loss value
        train_losses.append(loss.item())

        # save predictions
        train_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
    
    # update scheduler
    scheduler.step()

    # compute the average test loss
    train_loss = np.mean(train_losses)

    # Collect true labels into y_true
    y_true = np.array(train_true, dtype=np.float32)
    # Collect predictions into y_pred
    y_pred = np.array(train_predictions, dtype=np.float32)

    # Calculate accuracy as the average number of times y_true == y_pred
    accuracy = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true))])
    
    return train_loss, accuracy



#############################################
#### Test Function
#############################################

def test(model, test_loader, criterion, device):

    # put model in evaluation mode
    model.eval()

    # keep track of losses and predictions
    test_losses = []
    test_predictions = []
    test_true = []

    # We don't need gradients for validation, so wrap in 
    # no_grad to save memory

    with torch.no_grad():

        for batch, targets in test_loader:

            # Move the testing data to the GPU
            batch = batch.to(device, dtype=torch.float)
            test_true.extend(targets.numpy())
            targets = targets.to(device)

            # forward propagation
            predictions = model(batch)

            # calculate the loss
            loss = criterion(predictions, targets)

            # update running loss value
            test_losses.append(loss.item())

            # save predictions
            test_predictions.extend(predictions.argmax(dim=1).cpu().numpy())

    # compute the average test loss
    test_loss = np.mean(test_losses)

    # Collect true labels into y_true
    y_true = np.array(test_true, dtype=np.float32)
    # Collect predictions into y_pred
    y_pred = np.array(test_predictions, dtype=np.float32)

    # Calculate accuracy as the average number of times y_true == y_pred
    accuracy = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true))])

    return test_loss, accuracy



#############################################
#### Adversarial Test Function
#############################################

def test_adv(model, adv_loaders, criterion, device):

    # test adv phase
    test_adv_loss = []
    test_adv_accuracy = []
    for loader in adv_loaders:
        loss, accuracy = test(model, loader, criterion, device)
        test_adv_loss.append(loss)
        test_adv_accuracy.append(accuracy)
    
    return test_adv_loss, test_adv_accuracy


#############################################
#### Final Train Function
#############################################

def train(first_epoch, num_epochs, name, checkpoint_dir, model, train_loader, test_loader, adv_loaders, optimizer, scheduler):
    # set device
    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device);

    # Loss, Scheduler, Optimizer
    criterion = get_loss()

    # create log dictionary
    log_dict = {
        'current_test_loss'     : 999999,
        'current_test_accuracy' : 0,
        'current_test_epoch'    : 0,
        'best_test_loss'        : 999999,
        'best_test_accuracy'    : 0,
        'best_test_epoch'       : 0,
        'train_losses'          : [],
        'train_accuracies'      : [],
        'test_losses'           : [],
        'test_accuracies'       : [],
        'adv_losses'            : {
                                'severity_1'    : [],
                                'severity_2'    : [],
                                'severity_3'    : [],
                                'severity_4'    : [],
                                'severity_5'    : []
                                },
        'adv_accuracies'        : {
                                'severity_1'    : [],
                                'severity_2'    : [],
                                'severity_3'    : [],
                                'severity_4'    : [],
                                'severity_5'    : []
                                },
        'cls_reports'           : {
                                'train': {},
                                'test': {},
                                'severity_1': {},
                                'severity_2': {},
                                'severity_3': {},
                                'severity_4': {},
                                'severity_5': {}
                                } 
    }
    

    for epoch in range(first_epoch, first_epoch + num_epochs):
        # test print
        print(f'[{epoch:03d}] starting epoch')

        # training phase
        train_loss, train_accuracy = train_for_epoch(model, train_loader, criterion, optimizer, scheduler, device)

        # test phase
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        # adversarial test phase
        test_adv_loss, test_adv_accuracy = test_adv(model, adv_loaders, criterion, device)
        
        # print console log
        print(f'[{epoch:03d}] train loss: {train_loss:04f}', f'train accuracy: {train_accuracy:04f}')
        print(f'[{epoch:03d}] test loss: {test_loss:04f}', f'test accuracy: {test_accuracy:04f}')
        print(f'[{epoch:03d}] severity 1 loss: {test_adv_loss[0]:04f}', f'severity 1 accuracy: {test_adv_accuracy[0]:04f}')
        print(f'[{epoch:03d}] severity 2 loss: {test_adv_loss[1]:04f}', f'severity 2 accuracy: {test_adv_accuracy[1]:04f}')
        print(f'[{epoch:03d}] severity 3 loss: {test_adv_loss[2]:04f}', f'severity 3 accuracy: {test_adv_accuracy[2]:04f}')
        print(f'[{epoch:03d}] severity 4 loss: {test_adv_loss[3]:04f}', f'severity 4 accuracy: {test_adv_accuracy[3]:04f}')
        print(f'[{epoch:03d}] severity 5 loss: {test_adv_loss[4]:04f}', f'severity 5 accuracy: {test_adv_accuracy[4]:04f}\n')
        
        # updata log dictionary
        log_dict['train_losses'].append(train_loss)
        log_dict['train_accuracies'].append(train_accuracy)
        log_dict['test_losses'].append(test_loss)
        log_dict['test_accuracies'].append(test_accuracy)
        
        log_dict['current_test_loss'] = test_loss
        log_dict['current_test_accuracy'] = test_accuracy
        log_dict['current_test_epoch'] = epoch
        
        # update best accuracy
        if log_dict['current_test_accuracy'] > log_dict['best_test_accuracy']:
            log_dict['best_test_accuracy'] = log_dict['current_test_accuracy']
        
        # update adv losses/accuracies
        for i, (loss, accuracy) in enumerate(zip(test_adv_loss, test_adv_accuracy)):
            log_dict['adv_losses']['severity_{}'.format(i+1)].append(loss)
            log_dict['adv_accuracies']['severity_{}'.format(i+1)].append(accuracy)
        
        # Save current checkpoint
        checkpoint_name = name + '-CURRENT.pth'
        checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_name)
        save_checkpoint(optimizer, scheduler, model, epoch, checkpoint_filepath)
        
        # update best checkpoint
        if log_dict['current_test_loss'] < log_dict['best_test_loss']:
            log_dict['best_test_loss'] = log_dict['current_test_loss']
            log_dict['best_test_epoch'] = epoch

            checkpoint_name = name + '-BEST.pth'
            checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_name)
            save_checkpoint(optimizer, scheduler, model, epoch, checkpoint_filepath)

    # get classification reports
    print('[EVAL] Generating Classification Reports')
    log_dict['cls_reports']['train'] = get_classification_report(model, train_loader, device)
    log_dict['cls_reports']['test'] = get_classification_report(model, test_loader, device)
    for i, loader in enumerate(adv_loaders):
        log_dict['cls_reports']['severity_{}'.format(i+1)] = get_classification_report(model, loader, device)

    # save log dictionary
    log_name = name + '-LOG.pkl'
    log_filepath = os.path.join(checkpoint_dir, log_name)
    save_dict(log_filepath, log_dict)

    # save plot
    print('[EVAL] Generating Plots')
    plot_loss(log_dict, checkpoint_dir, figsize=10, linewidth=2, fonsize=15)
    plot_accuracy(log_dict, checkpoint_dir, figsize=10, linewidth=2, fonsize=15)
    
    return log_dict



#############################################
#### Evaluation Function
#############################################

def evaluate(name, test_loader, checkpoint_path, CLASSES):

    # get model name
    model_name = name.split('-')[0]

    # get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    model = get_model(model_name, len(CLASSES))

    # load checkpoint
    load_checkpoint(None, None, model, checkpoint_path)

    # Change to available device
    model.to(device);

    # put model in evaluation mode
    model.eval();
    
    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch, targets in test_loader:
            a = F.one_hot(targets, 11).numpy()
            y_true.append(a)

            # Move the testing data to the GPU
            batch = batch.to(device, dtype=torch.float)
            targets = targets.to(device)

            # forward propagation
            predictions = model(batch)
            predictions = F.softmax(predictions, dim=1)
            y_pred.append(predictions.cpu().numpy())

    y_true = np.concatenate(tuple(y_true), axis=0)
    y_pred = np.concatenate(tuple(y_pred), axis=0)
    
    return y_true, y_pred



#############################################
#### Save/Load Functions
#############################################

def save_checkpoint(optimizer, scheduler, model, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, scheduler, model, filename):
    print(filename)
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
    return epoch

def save_dict(path, log_dict):
    with open(path, 'wb') as f:
        pickle.dump(log_dict, f)

def load_dict(path):
    with open(path, 'rb') as f:
        log_dict = pickle.load(f)
    return log_dict



#############################################
#### Build Model Function
#############################################

def get_model(model_name, num_classes, pretrained=False):
    
    if model_name == 'Resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
    
        in_features = model.fc.in_features
        out_features = num_classes
        model.fc = nn.Linear(in_features, out_features)
        
        return model
    
    elif model_name == 'EfNetB0':

        # download pretrained model
        model = torchvision.models.efficientnet_b0(pretrained=pretrained)

        # define input and output features size
        in_features = model.classifier[1].in_features
        out_features = num_classes

        # replace the last layer
        model.classifier[1] = nn.Linear(in_features, out_features, bias=True)
        
        return model
    
    elif model_name == 'ViT_B16':

        # download pretrained model
        model = torchvision.models.vit_b_16(pretrained=pretrained)

        # define input and output features size
        in_features = model.heads.head.in_features
        out_features = num_classes

        # replace the last layer
        model.heads.head = nn.Linear(in_features, out_features, bias=True)
        
        return model



#############################################
#### Load Data Function
#############################################

def get_transforms(means, stds, adv_training=False, adv_dataset='gaussian_blur'):
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    if adv_training:
        if adv_dataset == 'gaussian_blur':
            train_transform = test_transform
        elif adv_dataset == 'saturate':
            train_transform = test_transform
        elif adv_dataset == 'elastic_transform':
            train_transform = test_transform
    else:
        train_transform = test_transform
    
    return train_transform, test_transform



#############################################
#### Load Data Function
#############################################

def load_dataset(train_batch_size, adv_training, adv_dataset, adv_data_path, adv_targets_path):

    train_data = CIFAR10(root='../datasets/CIFAR-10', train=True, download=True)

    # Calculate normalized mean and std for each channel
    means = train_data.data.mean(axis = (0,1,2)) / 255
    stds = train_data.data.std(axis = (0,1,2)) / 255

    # Define Transform
    train_transform, test_transform = get_transforms(means, stds, adv_training, adv_dataset)

    train_dataset = CIFAR10('../datasets/CIFAR-10', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10('../datasets/CIFAR-10', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, num_workers=8)

    # Define the classes
    CLASSES = train_dataset.classes

    # Define adv data/targets
    adv_data = np.load(adv_data_path)
    adv_targets = np.load(adv_targets_path)

    # Define adv datasets
    adv_dataset_sev_1 = ADVDataset(adv_data[:10000], adv_targets[0:10000], test_transform)
    adv_dataset_sev_2 = ADVDataset(adv_data[10000:20000], adv_targets[10000:20000], test_transform)
    adv_dataset_sev_3 = ADVDataset(adv_data[20000:30000], adv_targets[20000:30000], test_transform)
    adv_dataset_sev_4 = ADVDataset(adv_data[30000:40000], adv_targets[30000:40000], test_transform)
    adv_dataset_sev_5 = ADVDataset(adv_data[40000:], adv_targets[40000:], test_transform)

    adv_loader_sev_1 = DataLoader(adv_dataset_sev_1, batch_size=train_batch_size, shuffle=False, num_workers=8)
    adv_loader_sev_2 = DataLoader(adv_dataset_sev_2, batch_size=train_batch_size, shuffle=False, num_workers=8)
    adv_loader_sev_3 = DataLoader(adv_dataset_sev_3, batch_size=train_batch_size, shuffle=False, num_workers=8)
    adv_loader_sev_4 = DataLoader(adv_dataset_sev_4, batch_size=train_batch_size, shuffle=False, num_workers=8)
    adv_loader_sev_5 = DataLoader(adv_dataset_sev_5, batch_size=train_batch_size, shuffle=False, num_workers=8)

    adv_loaders = (adv_loader_sev_1, adv_loader_sev_2, adv_loader_sev_3, adv_loader_sev_4, adv_loader_sev_5)
    
    return train_loader, test_loader, adv_loaders, CLASSES



#############################################
#### Get Loss Function
#############################################

def get_loss():
    
    criterion = torch.nn.CrossEntropyLoss()
    return criterion



#############################################
#### Get Optimizer/Scheduler Function
#############################################

def get_optimizer_scheduler(model, initial_lr, step, gamma):
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    
    return optimizer, scheduler



#############################################
#### Plot Loss Single Training Run
#############################################

def plot_loss(log, checkpoint_dir, figsize=10, linewidth=2, fonsize=15):
    plt.figure(figsize=(figsize*2, figsize))
    color = iter(cm.Dark2(np.linspace(0, 1, len(log['adv_losses'])+2)))
    epochs = list(range(1, len(log['train_losses']) + 1))


    c = next(color)
    plt.plot(epochs, log['train_losses'], '-o', label='Train loss', lw=linewidth, c=c)
    c = next(color)
    plt.plot(epochs, log['test_losses'], '-o', label='Test loss', lw=linewidth, c=c)
    
    for i in range(len(log['adv_losses'])):
        loss = log['adv_losses']['severity_{}'.format(i+1)]
        c = next(color)
        plt.plot(epochs, loss, '-o', label='Adv Loss Sev {}'.format(i+1), lw=linewidth, c=c)

    plt.title('Loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend(loc="upper right", prop={'size':fonsize})
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, 'loss.png'), bbox_inches='tight')


#############################################
#### Plot Accuracy Single Training Run
#############################################

def plot_accuracy(log, checkpoint_dir, figsize=10, linewidth=2, fonsize=15):
    plt.figure(figsize=(figsize*2, figsize))
    color = iter(cm.Dark2(np.linspace(0, 1, len(log['adv_accuracies'])+2)))
    epochs = list(range(1, len(log['train_accuracies']) + 1))


    c = next(color)
    plt.plot(epochs, log['train_accuracies'], '-o', label='Train accuracy', lw=linewidth, c=c)
    c = next(color)
    plt.plot(epochs, log['test_accuracies'], '-o', label='Test accuracy', lw=linewidth, c=c)
    
    for i in range(len(log['adv_accuracies'])):
        accuracy = log['adv_accuracies']['severity_{}'.format(i+1)]
        c = next(color)
        plt.plot(epochs, accuracy, '-o', label='Adv accuracy Sev {}'.format(i+1), lw=linewidth, c=c)

    plt.title('Loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", prop={'size':fonsize})
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, 'accuracy.png'), bbox_inches='tight')



#############################################
#### Plot ROC Curve
#############################################

def plot_roc(name_list, fpr_tpr_list, auc_scores_list, figsize, linewidth, fontsize):
    plt.figure(figsize=(figsize, figsize))
    color = iter(cm.Dark2(np.linspace(0, 1, len(name_list))))
    model_name_list = [x.split('-')[1] for x in name_list]
    
    for i in range(len(model_name_list)):

        model_name = model_name_list[i]
        fpr, tpr = fpr_tpr_list[i]
        c = next(color)
        plt.plot(fpr, tpr, lw=linewidth, label=model_name + ' (auc={})'.format(auc_scores_list[i]), c=c)

    plt.plot([0, 1], [0, 1], color="navy", lw=linewidth, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontsize=fontsize)
    plt.title("Task 5 ROC Curve", fontsize=fontsize)
    plt.legend(loc="lower right", prop={'size':fontsize})
    plt.savefig('task5_ROC.png', bbox_inches='tight')
    plt.show()



#############################################
#### Plot Losses Multiple Runs
#############################################

def plot_loss_multi(name_list, losses_list, figw, figh, linewidth, fontsize, title, filename):
    plt.figure(figsize=(figw, figh))
    color = iter(cm.Dark2(np.linspace(0, 1, len(name_list))))
    model_name_list = [x.split('-')[1] for x in name_list]
    
    for i in range(len(model_name_list)):

        model_name = model_name_list[i]
        loss = losses_list[i]
        c = next(color)
        plt.plot(loss, lw=linewidth, label=model_name, c=c)

    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc="upper right", prop={'size':fontsize})
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
#############################################
#### Plot Multiclass ROC Curve
#############################################

def plot_multiclass_roc(name, fpr_tpr, auc_scores, figsize, linewidth, fontsize):

    plt.figure(figsize = (figsize*2, figsize))

    model_name = name.split('-')[0]
    fpr_dict, tpr_dict = fpr_tpr
    auc_scores_dict = auc_scores

    color = iter(cm.Dark2(np.linspace(0, 1, len(fpr_dict))))

    for key in auc_scores_dict:

        fpr = fpr_dict[key]
        tpr = tpr_dict[key]
        auc = auc_scores_dict[key]
        c = next(color)

        plt.plot(fpr, tpr, lw=linewidth, label = 'Class: {} (auc = {})'.format(key, auc), c=c)

    plt.plot([0, 1], [0, 1], color="navy", lw=linewidth, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontsize=fontsize)
    plt.legend(loc="lower right", prop={'size':fontsize})

    plt.title('{} ROC Curve'.format(model_name), fontsize=fontsize+10)
    plt.savefig('{}_ROC.png'.format(model_name), bbox_inches='tight')
    plt.show()


#############################################
#### Generate Classifiction Report
#############################################


def get_classification_report(model, test_loader, device):

    # put model in evaluation mode
    model.eval()

    # keep track of predictions
    test_predictions = []
    test_true = []

    # We don't need gradients for validation, so wrap in 
    # no_grad to save memory

    with torch.no_grad():

        for batch, targets in test_loader:

            # Move the testing data to the GPU
            batch = batch.to(device, dtype=torch.float)
            test_true.extend(targets.numpy())
            targets = targets.to(device)

            # forward propagation
            predictions = model(batch)
            # save predictions
            test_predictions.extend(predictions.argmax(dim=1).cpu().numpy())

    # Collect true labels into y_true
    y_true = np.array(test_true, dtype=np.float32)
    # Collect predictions into y_pred
    y_pred = np.array(test_predictions, dtype=np.float32)

    return classification_report(y_true, y_pred, output_dict=True)


#############################################
#### Custom Dataloader for npy Format
#############################################


class ADVDataset(Dataset):
    """ADV Dataset"""

    def __init__(self, data=None, targets=None, transform=None):

        self.data = data
        self.targets = targets
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, target