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
from PIL import Image
from torchvision import transforms
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

# Feel free to modify the arguments as you see fit
feature_map = {}
input_map = {}
def get_features(name):
    def hook(m, i, o):
        feature_map[name] = o.detach()
    return hook

def get_inputs(name):
    def hook(m, i, o):
        input_map[name] = i[0].detach()
    return hook

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
    parser.add_argument('--comp_subject_numb', type=int, default=8, help='The subject number for comparing the confusion matrix')
    parser.add_argument('--test_set', type=str, default='test', help='Test set to use (test or test_challenge)')
    parser.add_argument('--aug_train', action='store_true', help='Use augmented training data')
    parser.add_argument('--improved', action='store_true', help='whether or not to use the improved version')
    parser.add_argument('--load', action='store_true', help='whether or not to load saved model')
    parser.add_argument('--visualize_fm', action='store_true',
                        help='whether or not to visualize the feature map')
    parser.add_argument('--visualize_tSNE', action='store_true',
                        help='whether or not to visualize tSNE')


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
    improved_dir = 'improved' if args.improved else 'baseline'

    # Plot the training curve
    if args.dataset == 'Taiji':
        per_epoch_accs = []
        for i in range(1, args.num_subs+1):
            data = np.load(os.path.join(args.save_dir, args.dataset, args.fp_size, improved_dir, 'stats', f'sub_{i}.npz'))
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
        plt.savefig(os.path.join(args.save_dir, args.dataset, args.fp_size, improved_dir, 'plots', 'taiji_training_curve.png'))
        plt.close()
    else:
        if args.train:
            data = np.load(os.path.join(args.save_dir, args.dataset, args.test_set, improved_dir, 'stats', 'overall.npz'))
            per_epoch_accs = data['per_epoch_acc']
            fig = plt.figure()
            plt.plot(per_epoch_accs)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Wallpaper Training Accuracy ({args.test_set})')
            plt.savefig(os.path.join(args.save_dir, args.dataset, args.test_set, improved_dir, 'plots', 'wallpaper_training_curve.png'))
            plt.close()

        

def visualize(args, dataset, baseline_model=None, improved_model=None):
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
    improved_dir = 'improved' if args.improved else 'baseline'
    if dataset == 'Taiji':
        load_dir = os.path.join(args.save_dir, dataset, args.fp_size, improved_dir, 'stats')
        save_dir = os.path.join(args.save_dir, dataset, args.fp_size, improved_dir, 'plots')
    else:
        load_dir = os.path.join(args.save_dir, dataset, args.test_set, improved_dir, 'stats')
        save_dir = os.path.join(args.save_dir, dataset, args.test_set, improved_dir, 'plots')

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

    # Overall train/test acc and std averaged on all LOSO iterations
    if dataset == 'Taiji':
        overall_train_acc = overall_results['overall_train_acc']
        overall_train_acc_std = overall_results['overall_train_acc_std']
        overall_test_acc = overall_results['overall_test_acc']
        overall_test_acc_std = overall_results['overall_test_acc_std']
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        legend1 = ax.bar([0], [overall_train_acc], width=0.4, label='Accuracy', color='blue')
        legend2 = ax2.bar([0.4], [overall_train_acc_std], width=0.4, label='Standard Deviation', color='orange')
        ax.bar([1], [overall_test_acc], width=0.4, label='Accuracy', color='blue')
        ax2.bar([1.4], [overall_test_acc_std], width=0.4, label='Standard Deviation', color='orange')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Standard Deviation')
        ax.set_xlabel('Training or Testing')
        ax.set_title('Training/Testing overall accuracy & std, averaged on all iterations')
        ax.set_xticks([0.2, 1.2])
        ax.set_xticklabels(['Training', 'Testing'])
        ax.legend(handles=[legend1, legend2], loc='upper center')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'averaged overall acc and std.png'))
        plt.close()

    # Overall train/test acc and std for Wallpaper
    if dataset == 'Wallpaper':
        sub_class_train = overall_results['sub_class_train']
        sub_class_test = overall_results['sub_class_test']
        overall_train_acc = np.mean(sub_class_train)
        overall_train_acc_std = np.std(sub_class_train)
        overall_test_acc = np.mean(sub_class_test)
        overall_test_acc_std = np.std(sub_class_test)

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.bar([0], [overall_train_acc], width=0.4, label='Accuracy', color='blue')
        ax2.bar([1], [overall_train_acc_std], width=0.4, label='Standard Deviation', color='orange')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Standard Deviation')
        ax.set_xlabel('Overall accuracy and standard deviation')
        ax.set_title('Overall training accuracy & std')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Accuracy', 'Standard Deviation'])
        ax.legend(loc='upper center')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training overall accuracy and std.png'))
        plt.close()

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.bar([0], [overall_test_acc], width=0.4, label='Accuracy', color='blue')
        ax2.bar([1], [overall_test_acc_std], width=0.4, label='Standard Deviation', color='orange')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Standard Deviation')
        ax.set_xlabel('Overall accuracy and standard deviation')
        ax.set_title('Overall testing accuracy & std')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Accuracy', 'Standard Deviation'])
        ax.legend(loc='upper center')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'testing overall accuracy and std.png'))
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

    # Overall per class testing data std (Taiji only)
    if dataset == 'Taiji':
        sub_class_test = overall_results['sub_class_test']
        std_per_class_test = np.std(sub_class_test, axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(num_classes), std_per_class_test)
        ax.set_ylabel('Accuracy Standard Deviation')
        ax.set_xlabel('Class')
        ax.set_title(f'{dataset} Overall per class testing accuracy standard deviation')
        ax.set_xticks(np.arange(num_classes))
        ax.set_xticklabels(label_names, rotation='vertical')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'std_per_class_test.png'))

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

    # Testing confusion matrix of specific subject with sklearns display (Taiji only)
    if dataset == 'Taiji':
        fig, ax = plt.subplots(figsize=(10, 10))
        comp_test_conf_mat = overall_results['comp_test_conf_mat']
        comp_subj_numb = overall_results['comp_subj_numb']
        disp = ConfusionMatrixDisplay(comp_test_conf_mat, display_labels=label_names)
        disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
        disp.ax_.get_images()[0].set_clim(0, 1)
        ax.set_title(f'{dataset} Testing Confusion Matrix of Subject {comp_subj_numb}')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comp_test_conf_mat.png'))

    # Visualize t-SNE
    if args.visualize_tSNE:
        save_tSNE_path = os.path.join(args.save_dir, 'Wallpaper', 't-SNE')
        if not os.path.exists(save_tSNE_path):
            os.mkdir(save_tSNE_path)

        # run through test subset to retrieve visualization data
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        test_dataset = ImageFolder(os.path.join(args.data_root, 'Wallpaper', args.test_set), transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=10*args.batch_size, shuffle=False)

        device = torch.device('cpu')
        # load model parameters
        model_save_path = os.path.join(args.save_dir, 'Wallpaper', args.test_set,
                                       improved_dir, 'model', 'model.pt')
        checkpoint = torch.load(model_save_path)
        if args.improved:
            model = improved_model(input_channels=1, img_size=args.img_size, num_classes=num_classes)
        else:
            model = baseline_model(input_channels=1, img_size=args.img_size, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # register hooks to retrieve conv and fully connected layer output
        model.fc_2.register_forward_hook(get_features('fcn'))

        # run through one batch size and plot the t-SNE
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            model(data)

            tSNE_embed = TSNE().fit_transform(feature_map['fcn'])
            x_coor = [embed[0] for embed in tSNE_embed]
            y_coor = [embed[1] for embed in tSNE_embed]
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.scatter(x_coor, y_coor)
            ax.set_ylabel('dim 2')
            ax.set_xlabel('dim 1')
            ax.set_title('t-SNE visualization on the last fully connected layer (640 data points)')
            fig.tight_layout()
            plt.savefig(os.path.join(save_tSNE_path, 't-SNE.png'))
            break

    # Visualize feature map of chosen convolutional layer with 17 images from 17 wallpaper groups and perform t-SNE
    if args.visualize_fm:
        # create visualization image folder for testing
        target_image_path = os.path.join(args.data_root, 'Wallpaper', 'visualization')
        if not os.path.exists(target_image_path):
            image_set_path = os.path.join(args.data_root, 'Wallpaper', 'test')

            for sub_dir in os.listdir(image_set_path):
                image_dir = os.path.join(image_set_path, sub_dir)
                target_image_path = os.path.join(args.data_root, 'Wallpaper', 'visualization', sub_dir)
                os.makedirs(target_image_path)
                for image in os.listdir(image_dir):
                    orig_image_path = os.path.join(image_dir, image)
                    shutil.copy(orig_image_path, target_image_path)
                    break

        if dataset == 'Wallpaper':
            device = torch.device('cpu')

            # load model parameters
            model_save_path = os.path.join(args.save_dir, 'Wallpaper', args.test_set,
                                           improved_dir, 'model', 'model.pt')
            checkpoint = torch.load(model_save_path)
            if args.improved:
                model = improved_model(input_channels=1, img_size=args.img_size, num_classes=num_classes)
            else:
                model = baseline_model(input_channels=1, img_size=args.img_size, num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            # register hooks to retrieve conv and fully connected layer output
            model.first_conv.register_forward_hook(get_features('conv'))

            # pass images into CNN
            test_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            visualize_dataset = ImageFolder(os.path.join(args.data_root, 'Wallpaper', 'visualization'), transform=test_transform)
            visualize_loader = DataLoader(visualize_dataset, batch_size=args.batch_size, shuffle=False)
            for batch_idx, (data, target) in enumerate(visualize_loader):
                data, target = data.to(device), target.to(device)
                model(data)

            # create a mapping from index to image directory
            dir2idx_dict = visualize_dataset.class_to_idx
            idx2dir_dict = {}
            for dir_name, idx in dir2idx_dict.items():
                idx2dir_dict[idx] = dir_name

            # feature maps visualization
            save_fm_path = os.path.join(args.save_dir, 'Wallpaper', 'conv_layer_feature_map')
            if not os.path.exists(save_fm_path):
                os.mkdir(save_fm_path)

            act = feature_map['conv'].squeeze()
            fig, ax = plt.subplots(2)
            ax[0].imshow(act[0][0])
            for img_idx in range(act.size(0)):
                fig, ax = plt.subplots(6, 6)
                for idx in range(act.size(1)):
                    ax[idx // 6][idx % 6].imshow(act[img_idx][idx])
                    ax[idx // 6][idx % 6].axis('off')
                # remove unused subplots
                for idx in range(act.size(1), 36):
                    fig.delaxes(ax[idx // 6][idx % 6])

                fig.tight_layout()
                plt.savefig(os.path.join(save_fm_path, idx2dir_dict[img_idx] + '.png'))
                plt.close()

    return

