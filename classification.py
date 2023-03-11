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
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from util import *
from models import *

def test(model, test_loader, device, criterion):
    """
    Test the model.
    Args:
        model (torch.nn.Module): Model to test.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device): Device to use (cuda or cpu).
        criterion (torch.nn.Module): Loss function to use.
    Returns:
        test_loss (float): Average loss on the test set.
        test_acc (float): Average accuracy on the test set.
        preds (numpy array): Class predictions.
        targets (numpy array): Target values.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        preds = []
        targets = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return test_loss, test_acc, preds, targets

#  Note log_interval doesn't actually log to a file but is used for printing. This can be changed if you want to log to a file.
def train(model, train_loader, optimizer, criterion, epochs, 
          log_interval, device):
    """
    Train the model and periodically log the loss and accuracy.
    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        criterion (torch.nn.Module): Loss function to use.
        epochs (int): Number of epochs to train for.
        log_interval (int): Print loss every log_interval epochs.
        device (torch.device): Device to use (cuda or cpu).
    Returns:
        per_epoch_loss (list): List of loss values per epoch.
        per_epoch_acc (list): List of accuracy values per epoch.
    """
    model.train()
    per_epoch_loss = []
    per_epoch_acc = []
    for epoch in range(epochs):
        train_loss = 0
        preds = []
        targets = []
        correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Get the accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Save the predictions and targets if it's the last epoch
            if epoch == epochs - 1:
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        per_epoch_acc.append(train_acc)
        if epoch % log_interval == 0:
            print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch, train_loss, train_acc))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return model, np.array(per_epoch_loss), np.array(per_epoch_acc), preds, targets   

def wallpaper_main(args):
    """
    Main function for training and testing the wallpaper classifier.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_classes = 17
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load the wallpaper dataset
    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO: Augment the training data given the transforms in the assignment description.
    if args.aug_train:
        raise NotImplementedError


    # Compose the transforms that will be applied to the images. Feel free to adjust this.
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    train_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=transform)
    test_dataset = ImageFolder(os.path.join(data_root, args.test_set), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    print(f"Training on {len(train_dataset)} images, testing on {len(test_dataset)} images.")
    # Initialize the model, optimizer, and loss function
    model = CNN(input_channels=1, img_size=args.img_size, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train + test the model
    model, per_epoch_loss, per_epoch_acc, train_preds, train_targets = train(model, train_loader, optimizer, criterion, args.num_epochs, 
                                                                             args.log_interval, device )
    test_loss, test_acc, test_preds, test_targets = test(model, test_loader, device, criterion)

    # Get stats 
    classes_train, overall_train_mat = get_stats(train_preds, train_targets, num_classes)
    classes_test, overall_test_mat = get_stats(test_preds, test_targets, num_classes)


    print(f'\n\nTrain accuracy: {per_epoch_acc[-1]*100:.3f}')
    print(f'Test accuracy: {test_acc*100:.3f}')

    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats')):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats'))
    overall_file_name = os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats', 'overall.npz')

    np.savez(overall_file_name, classes_train=classes_train, overall_train_mat=overall_train_mat, 
                classes_test=classes_test, overall_test_mat=overall_test_mat, 
                per_epoch_loss=per_epoch_loss, per_epoch_acc=per_epoch_acc, 
                test_loss=test_loss, test_acc=test_acc)

    # Note: The code does not save the model but you may do so if you choose with the args.save_model flag.

def taiji_main(args):
    """
    Main function for training and testing the taiji classifier.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_subs = args.num_subs
    num_forms = 46 # Number of taiji forms, hardcoded :p
    sub_train_acc = np.zeros(num_subs)
    sub_class_train = np.zeros((num_subs, num_forms))
    sub_test_acc = np.zeros(num_subs)
    sub_class_test = np.zeros((num_subs, num_forms))
    overall_train_mat = np.zeros((num_forms, 1))
    overall_test_mat = np.zeros((num_forms, 1))
    improved_dir = 'improved' if args.improved else 'baseline'

    if not os.path.exists(os.path.join(args.save_dir, 'Taiji', args.fp_size)):
        os.makedirs(os.path.join(args.save_dir, 'Taiji', args.fp_size))

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("GPU is not available, use CPU instead.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # For each subject LOSO
    for i in range(num_subs):
        print('\n\nTraining subject: {}'.format(i+1))
        train_data = TaijiData(data_dir='data', subject=i+1, split='train', fp_size=args.fp_size)
        test_data = TaijiData(data_dir ='data', subject=i+1, split='test', fp_size=args.fp_size)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        model = MLP(input_dim=train_data.data_dim, hidden_dim=1024, output_dim=num_forms, improved=args.improved).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Train + test the model
        model, train_losses, per_epoch_train_acc, train_preds, train_targets \
                        = train(model, train_loader, optimizer, criterion, args.num_epochs, args.log_interval, device)
        test_loss, test_acc, test_pred, test_targets = test(model, test_loader, device, criterion)

        # Print accs to three decimal places
        sub_train_acc[i] = per_epoch_train_acc[-1]
        sub_test_acc[i] = test_acc
        print(f'Subject {i+1} Train Accuracy: {per_epoch_train_acc[-1]*100:.3f}')
        print(f'Subject {i+1} Test Accuracy: {test_acc*100:.3f}')
        
        # Save all stats (you can save the model if you choose to)
        if not os.path.exists(os.path.join(args.save_dir, 'Taiji', args.fp_size, improved_dir, 'stats')):
            os.makedirs(os.path.join(args.save_dir, 'Taiji', args.fp_size, improved_dir, 'stats'))

        sub_file = os.path.join(args.save_dir, 'Taiji', args.fp_size, improved_dir, 'stats', 'sub_{}.npz'.format(i+1))
        classes_train, conf_mat_train = get_stats(train_preds, train_targets, num_forms)
        classes_test, conf_mat_test = get_stats(test_pred, test_targets, num_forms)
        sub_class_train[i, :] = classes_train
        sub_class_test[i, :] = classes_test
        overall_train_mat = overall_train_mat + (1/num_subs) * conf_mat_train 
        overall_test_mat = overall_test_mat + (1/num_subs) * conf_mat_test 
        np.savez(sub_file, train_losses=train_losses, per_epoch_acc=per_epoch_train_acc, test_acc=test_acc,
                 conf_mat_train=conf_mat_train, conf_mat_test=conf_mat_test)

        # Note: the code does not save the model, but you may choose to do so with the arg.save_model flag
        if args.save_model:
            if not os.path.exists(os.path.join(args.save_dir, 'Taiji', args.fp_size, improved_dir, 'model')):
                os.makedirs(os.path.join(args.save_dir, 'Taiji', args.fp_size, improved_dir, 'model'))

            model_save_file = os.path.join(args.save_dir, 'Taiji', args.fp_size,
                                           improved_dir, 'model', 'state_dict_{}.pt'.format(i+1))
            torch.save(model.state_dict(), model_save_file)
        
    # Save overall stats
    overall_train_acc = np.mean(sub_train_acc)
    overall_test_acc = np.mean(sub_test_acc)
    print(f"\n\nOverall Train Accuracy: {overall_train_acc:.3f}")
    print(f"Overall Test Accuracy: {overall_test_acc:.3f}")

    overall_file_name = os.path.join(args.save_dir, 'Taiji', args.fp_size, improved_dir, 'stats', 'overall.npz')
    np.savez(overall_file_name, sub_train_acc = sub_train_acc, sub_class_train=sub_class_train,
             sub_test_acc=sub_test_acc, sub_class_test=sub_class_test, overall_train_mat=overall_train_mat, overall_test_mat=overall_test_mat)


if __name__ == '__main__':
    args = arg_parse()

    if args.dataset == 'Wallpaper':
        if args.train:
            wallpaper_main(args)
        visualize(args, dataset='Wallpaper')
        plot_training_curve(args)
    else: 
        if args.train:
            taiji_main(args)
        visualize(args, dataset='Taiji')
        plot_training_curve(args)

