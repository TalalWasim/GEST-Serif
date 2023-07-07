from operator import ne
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.hub import _get_torch_home

from tqdm import tqdm
import numpy as np
import random
import os
import shutil
import pickle
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, cm

from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from torchvision.models import get_model




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
    for batch, targets in tqdm(train_loader):

        # Move the training data to the GPU
        batch = batch.to(device, dtype=torch.float)
        train_true.extend(targets.numpy())
        targets = targets.to(device)

        # clear previous gradient computation
        optimizer.zero_grad()

        # forward propagation
        predictions = model(batch)

        # calculate the loss
        loss = criterion(predictions, targets)

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
    train_accuracy = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true))])
    
    return train_loss, train_accuracy



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

        for batch, targets in tqdm(test_loader):

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
    test_accuracy = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true))])

    return test_loss, test_accuracy



#############################################
#### Evaluation Function
#############################################

def evaluate(name, test_loader, checkpoint_path, CLASSES):

    # get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    model = create_model(name, num_classes=len(CLASSES))

    # load checkpoint
    load_checkpoint(None, None, model, checkpoint_path)

    # Change to available device
    model.to(device);

    # put model in evaluation mode
    model.eval();
    
    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch, targets in tqdm(test_loader):
            a = F.one_hot(targets, len(CLASSES)).numpy()
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
#### Final Train Function
#############################################

def train(args):
    # set device
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

    # Build dataloaders
    train_loader, test_loader, CLASSES = load_dataset(args.train_folder,
                                                      args.test_folder,
                                                      train_batch_size=args.batch_size,
                                                      resize_size=args.resize_size,
                                                      crop_size=args.crop_size,
                                                      num_workers=args.num_workers)

    # get class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(train_loader.dataset.targets),
                                                      y=np.array(train_loader.dataset.targets))
    class_weights=torch.tensor(class_weights, dtype=torch.float).to(device)

    # create model
    model = create_model(args.model, num_classes=len(CLASSES), image_size=args.crop_size)
    model = model.to(device);

    # Loss, Scheduler, Optimizer
    criterion = get_loss(class_weights)
    optimizer, scheduler = get_optimizer_scheduler(model, args.initial_lr, args.step, args.gamma)

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
        'cls_reports'           : {
                                'train': {},
                                'test': {},
                                } 
    }
    

    for epoch in range(1, 1 + args.epochs):
        # start print
        print(f'[{epoch:03d}] starting epoch')

        # training phase
        print(f'[{epoch:03d}] starting training phase')
        train_loss, train_accuracy = train_for_epoch(model, train_loader, criterion, optimizer, scheduler, device)

        # test phase
        print(f'[{epoch:03d}] starting testing phase')
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
        # print console log
        print(f'[{epoch:03d}] train loss: {train_loss:04f}', f'train accuracy: {train_accuracy:04f}')
        print(f'[{epoch:03d}] test loss: {test_loss:04f}', f'test accuracy: {test_accuracy:04f}\n')
        
        # update log dictionary
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
        
        # Save current checkpoint
        checkpoint_name = 'checkpoint-LAST.pth'
        checkpoint_filepath = os.path.join(args.checkpoint_dir, checkpoint_name)
        save_checkpoint(optimizer, scheduler, model, epoch, checkpoint_filepath)
        
        # update best checkpoint
        if log_dict['current_test_loss'] < log_dict['best_test_loss']:
            log_dict['best_test_loss'] = log_dict['current_test_loss']
            log_dict['best_test_epoch'] = epoch

            checkpoint_name = 'checkpoint-BEST.pth'
            checkpoint_filepath = os.path.join(args.checkpoint_dir, checkpoint_name)
            save_checkpoint(optimizer, scheduler, model, epoch, checkpoint_filepath)

    # get classification reports
    print('[EVAL] Generating Classification Reports')
    log_dict['cls_reports']['train'] = get_classification_report(model, train_loader, device)
    log_dict['cls_reports']['test'] = get_classification_report(model, test_loader, device)

    # save log dictionary
    log_name = 'LOG.pkl'
    log_filepath = os.path.join(args.checkpoint_dir, log_name)
    save_dict(log_filepath, log_dict)

    # save plot
    print('[EVAL] Generating Plots')
    plot_loss(log_dict, args.checkpoint_dir, figsize=10, linewidth=2, fontsize=15)
    plot_accuracy(log_dict, args.checkpoint_dir, figsize=10, linewidth=2, fontsize=15)
    
    return log_dict



############################################
#### Save/Load Functions
############################################

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
#### Create Model
#############################################

def create_model(name, num_classes, image_size=None):

    if 'efficientnet' in name:
        model = get_model(name, pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes, bias=True)
    elif 'resnet' in name:
        model = get_model(name, pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif 'vit' in name:
        model = get_model(name, pretrained=True)

        if image_size is not None:
            torch_home = _get_torch_home()
            if 'b_16' in name:
                model = get_model(name, image_size=image_size)
                ckpt = torch.load(os.path.join(torch_home,'hub/checkpoints/vit_b_16-c867db91.pth'), map_location='cpu')
                del ckpt['encoder.pos_embedding']
                model.load_state_dict(ckpt, strict=False)
            elif 'b_32' in name:
                model = get_model(name, image_size=image_size)
                ckpt = torch.load(os.path.join(torch_home,'hub/checkpoints/vit_b_32-d86f8d99.pth'), map_location='cpu')
                del ckpt['encoder.pos_embedding']
                model.load_state_dict(ckpt, strict=False)

        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError('Model {} not implemented in create_model function'.format(name))

    return model



#############################################
#### Get Transforms
#############################################

def get_transforms(resize_size=600, crop_size=448, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):


    train_transform=transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BICUBIC),
                                    transforms.RandomCrop((crop_size, crop_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform=transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BICUBIC),
                                    transforms.CenterCrop((crop_size, crop_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transform, test_transform



#############################################
#### Load Data
#############################################

def load_dataset(root_train, root_test, train_batch_size, resize_size=600, crop_size=448, num_workers=8):

    # Define Transform
    train_transform, test_transform = get_transforms(resize_size=resize_size, crop_size=crop_size)

    #Load image
    dataset_train = ImageFolder(root=root_train, transform=train_transform)
    dataset_test = ImageFolder(root=root_test, transform=test_transform)
    

    # Create DataLoaders
    train_loader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset_test, batch_size=train_batch_size, shuffle=False, num_workers=num_workers)
    

    # Define the classes
    CLASSES = dataset_train.classes
    
    return train_loader, test_loader, CLASSES



#############################################
#### Get Loss Function
#############################################

def get_loss(class_weights=None):

    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    else:
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

def plot_loss(log, checkpoint_dir, figsize=10, linewidth=2, fontsize=15):
    plt.figure(figsize=(figsize*2, figsize))
    color = iter(cm.Dark2(np.linspace(0, 1, 2)))
    epochs = list(range(1, len(log['train_losses']) + 1))


    c = next(color)
    plt.plot(epochs, log['train_losses'], '-o', label='Train loss', lw=linewidth, c=c)
    c = next(color)
    plt.plot(epochs, log['test_losses'], '-o', label='Test loss', lw=linewidth, c=c)
    

    plt.title('Loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend(loc="upper right", prop={'size':fontsize})
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, 'loss.png'), bbox_inches='tight')



#############################################
#### Plot Accuracy Single Training Run
#############################################

def plot_accuracy(log, checkpoint_dir, figsize=10, linewidth=2, fontsize=15):
    plt.figure(figsize=(figsize*2, figsize))
    color = iter(cm.Dark2(np.linspace(0, 1, 2)))
    epochs = list(range(1, len(log['train_accuracies']) + 1))


    c = next(color)
    plt.plot(epochs, log['train_accuracies'], '-o', label='Train accuracy', lw=linewidth, c=c)
    c = next(color)
    plt.plot(epochs, log['test_accuracies'], '-o', label='Test accuracy', lw=linewidth, c=c)

    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", prop={'size':fontsize})
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, 'accuracy.png'), bbox_inches='tight')



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

        for batch, targets in tqdm(test_loader):

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
#### Print Classification Report
#############################################

def print_classification_report(data_dict):
    """Build a text report showing the main classification metrics.
    Read more in the :ref:`User Guide <classification_report>`.
    Parameters
    ----------
    report : string
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy otherwise.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    """

    non_label_keys = ["accuracy", "macro avg", "weighted avg"]
    y_type = "binary"
    digits = 2

    target_names = [
        "%s" % key for key in data_dict.keys() if key not in non_label_keys
    ]

    # labelled micro average
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary")

    headers = ["precision", "recall", "f1-score", "support"]
    p = [data_dict[l][headers[0]] for l in target_names]
    r = [data_dict[l][headers[1]] for l in target_names]
    f1 = [data_dict[l][headers[2]] for l in target_names]
    s = [data_dict[l][headers[3]] for l in target_names]

    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    longest_last_line_heading = "weighted avg"
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)
    report += "\n"

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        if line_heading == "accuracy":
            avg = [data_dict[line_heading], sum(s)]
            row_fmt_accuracy = "{:>{width}s} " + \
                    " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + \
                    " {:>9}\n"
            report += row_fmt_accuracy.format(line_heading, "", "",
                                              *avg, width=width,
                                              digits=digits)
        else:
            avg = list(data_dict[line_heading].values())
            report += row_fmt.format(line_heading, *avg,
                                     width=width, digits=digits)
    return report