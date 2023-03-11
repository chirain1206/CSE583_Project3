'''
Start code for Project 3
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name:
    PSU Email ID:
    Description: (A short description of what each of the functions you've written does.).
}
'''
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# Feel free to modify the arguments as you see fit
def arg_parse():
    """
    Parses the arguments.
    Returns:
        parser (argparse.ArgumentParser): Parser with arguments.
    """
    parser = argparse.ArgumentParser()
    # General 
    parser.add_argument('--dataset', type=str, default='Wallpaper', help='Dataset to use (Taiji or Wallpaper)')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--data_root', type=str, default='data', help='Directory to save results')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--log_interval', type=int, default=1, help='Print loss every log_interval epochs, feel free to change')
    parser.add_argument('--train' , action='store_true', help='Train the model')
    parser.add_argument('--save_model', action='store_true', help='Save the model')
    # Taji specific
    parser.add_argument('--num_subs', type=int, default=10, help='Number of subjects to train and test on')
    parser.add_argument('--fp_size', type=str, default='lod4', help='Size of the fingerprint to use (lod4 or full)')
    # Wallpapers specific
    parser.add_argument('--img_size', type=int, default=128, help='Size of image to be resized to')
    parser.add_argument('--test_set', type=str, default='test', help='Test set to use (test or test_challenge)')
    parser.add_argument('--aug_train', action='store_true', help='Use augmented training data')
     

    return parser.parse_args()

def get_stats(preds, targets, num_classes):
    """
    Calculates the prediction stats.
    Args:
        preds (numpy array): Class predictions.
        targets (numpy array): Target values.
        num_classes (int): Number of classes.
    Returns:
        class_correct (numpy array): Array of the number of correct predictions for each class.
        conf_mat (numpy array): Confusion matrix.
    """
    # Get conf matrix
    gt_label = np.arange(num_classes)
    conf_mat = confusion_matrix(targets, preds, labels=gt_label, normalize='true')
    class_correct = np.diag(conf_mat)

    return class_correct, conf_mat

def prep_data(data, labels):
    """
    Preprocess the data and labels by turning them into tensors and normalizing
    Args:
        data: [N, D] tensor of data
        labels: [N, 1] tensor of labels
    Returns:
        data: [N, D] tensor of data
        labels: [N, 1] tensor of labels
    """
    data = torch.from_numpy(data).float()
    data = F.normalize(data, p=2, dim=1)
    labels = torch.from_numpy(labels).long()
    return data, labels

class TaijiData(Dataset):
    def __init__(self, data_dir, subject=1, split='train', fp_size='lod4'):
        """
        Args:
            data_dir (string): Directory the data.npz file.
            subject (int): Subject number for LOSO data split
            split (string): train or test
        """
        self.data_dir = data_dir
        self.subject = subject
        self.split = split
        self.fp_size = fp_size
        self.data_dim = None
        self.data = None # [N, D] tensor of data
        self.labels = None # [N, 1] tensor of labels
        self.load_data()

    def load_data(self):
        """
        Load the data and labels in self.data_dir. Note the fp_size argument to control the foot pressure map size.
        """
        if self.fp_size == 'full':
            taiji_data = np.load(os.path.join(self.data_dir, 'Taiji_data_full_fp.npz'))
        else:
            taiji_data = np.load(os.path.join(self.data_dir, 'Taiji_data_lod4_fp.npz'))
        data = taiji_data['data']
        data[np.isnan(data)] = 0.
        self.data_dim = data.shape[1]
        labels = taiji_data['labels']
        sub_info = taiji_data['sub_info']

        # Get the indices of the test and train data
        if self.split == 'train':
            train_inds = np.where(sub_info[:, 0] != self.subject)[0]
            self.data, self.labels = prep_data(data[train_inds, :], labels[train_inds])
        else:
            test_inds = np.where(sub_info[:, 0] == self.subject)[0]
            self.data, self.labels = prep_data(data[test_inds, :], labels[test_inds])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def plot_training_curve(args):
    """
    Plot the training curve
    Args:
        args: Arguments
    """
    # Plot the training curve
    if args.dataset == 'Taiji':
        per_epoch_accs = []
        for i in range(1, args.num_subs+1):
            data = np.load(os.path.join(args.save_dir, args.dataset, args.fp_size, 'stats', f'sub_{i}.npz'))
            per_epoch_accs.append(data['per_epoch_acc'])
        per_epoch_accs = np.array(per_epoch_accs)
        # Plot each subject training acc curve
        fig = plt.figure()
        for i in range(args.num_subs):
            # Label each sub
            plt.plot(per_epoch_accs[i], label=f'Sub {i+1}', alpha=0.5, linewidth=2, marker='o')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Taiji Subject Training Accuracy ({args.fp_size} fp)')
        plt.savefig(os.path.join(args.save_dir, args.dataset, args.fp_size, 'plots', 'taiji_training_curve.png'))
        plt.close()
    else:
        data = np.load(os.path.join(args.save_dir, args.dataset, args.test_set, 'stats', 'overall.npz'))
        per_epoch_accs = data['per_epoch_acc']
        fig = plt.figure()
        plt.plot(per_epoch_accs)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Wallpaper Training Accuracy ({args.test_set})')
        plt.savefig(os.path.join(args.save_dir, args.dataset, args.test_set, 'plots', 'wallpaper_training_curve.png'))
        plt.close()

        

def visualize(args, dataset):
    """
    Visualize the results
    Args:
        args: Arguments
        dataset: Dataset to use (Taiji or Wallpaper)
    """
    if args.dataset == 'Taiji':
        num_classes = 46
    else:
        num_classes = 17 

    # Get label names
    if args.dataset == 'Taiji':
        if args.fp_size == 'full':
            data = np.load(os.path.join(args.data_root, 'Taiji_data_full_fp.npz'))
        else:
            data = np.load(os.path.join(args.data_root, 'Taiji_data_lod4_fp.npz'))
        label_names = np.arange(0, 46) # The form names are pretty long so its doesn't work great in the plots
    else:
        label_names = os.listdir(os.path.join(args.data_root, 'Wallpaper', 'train'))

    # Save and load dirs
    if dataset == 'Taiji':
        load_dir = os.path.join(args.save_dir, dataset, args.fp_size, 'stats')
        save_dir = os.path.join(args.save_dir, dataset, args.fp_size, 'plots')
    else:
        load_dir = os.path.join(args.save_dir, dataset, args.test_set, 'stats')
        save_dir = os.path.join(args.save_dir, dataset, args.test_set, 'plots')

    overall_file = os.path.join(load_dir, 'overall.npz')
    overall_results = np.load(overall_file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Overall subject training and testing rates as a grouped bar chart
    if dataset == 'Taiji':
        sub_train_acc = overall_results['sub_train_acc']
        sub_test_acc = overall_results['sub_test_acc']
        fig, ax = plt.subplots()
        ax.bar(np.arange(10), sub_train_acc, width=0.35, label='Training')
        ax.bar(np.arange(10)+0.35, sub_test_acc, width=0.35, label='Testing')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Subject')
        ax.set_title('Subject-wise training and testing accuracies')
        # Make the x-axis labels start from 1
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels(np.arange(1, 10+1))
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'subject_wise_acc.png'))
        plt.close()


    # Overall per class training data. Tilt the x-axis labels by 45 degrees
    overall_train_mat = overall_results['overall_train_mat']
    overall_per_class_train = overall_train_mat.diagonal()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(num_classes), overall_per_class_train)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Class')
    ax.set_title(f'{dataset} Overall per class training accuracy')
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(label_names, rotation='vertical')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_per_class_train.png'))
    plt.close()

    # Overall per class testing data
    overall_test_mat = overall_results['overall_test_mat']
    overall_per_class_test = overall_test_mat.diagonal()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(num_classes), overall_per_class_test)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Class')
    ax.set_title(f'{dataset} Overall per class testing accuracy')
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(label_names, rotation='vertical')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_per_class_test.png'))

    # Overall training confusion matrix with sklearns display
    fig, ax = plt.subplots(figsize=(10, 10))
    overall_train_mat = overall_results['overall_train_mat']
    disp = ConfusionMatrixDisplay(overall_train_mat, display_labels=label_names, )
    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
    disp.ax_.get_images()[0].set_clim(0, 1)
    ax.set_title(f'{dataset} Overall training confusion matrix')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_train_conf_mat.png'))

    # Overall testing confusion matrix with sklearns display
    fig, ax = plt.subplots(figsize=(10, 10))
    overall_test_mat = overall_results['overall_test_mat']
    disp = ConfusionMatrixDisplay(overall_test_mat, display_labels=label_names)
    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
    disp.ax_.get_images()[0].set_clim(0, 1)
    ax.set_title(f'{dataset} Overall Testing Confusion Matrix')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_test_conf_mat.png'))
    return

