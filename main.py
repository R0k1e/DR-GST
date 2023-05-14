'''
Rechoose pseudo labels every stage
'''
import argparse
import numpy as np
import torch
import torch.optim as optim
import random
from utils import accuracy
from utils import *
from utils_plot import *
import torch.nn as nn
import os


global result
result = []
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Cora",
                    help='dataset for training')
parser.add_argument('--labelrate', type=int, required=True, default=20)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--stage', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.53)
parser.add_argument('--beta', type=float, default=1/3,
                    help='coefficient for weighted CE loss')
parser.add_argument('--drop_method', type=str, required=True, default='dropout')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--droprate', type=float, default=0.5,
                    help='Droprate for MC-Dropout')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, seed):
    sign = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
     # Get the number of classes
    nclass = labels.max().item() + 1
     # Get the model
    model = get_models(args, features.shape[1], nclass, g=g)
     # Create the optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
     # Move the model to the device
    model.to(device)
     # Initialize the best loss and bad counter
    best, bad_counter = 100, 0
     # Train the model for the given number of epochs
    for epoch in range(args.epochs):
        # Set the model to training mode
        model.train()
         # Zero the gradients
        optimizer.zero_grad()
         # Get the output of the model
        output = model(features, adj)
         # Softmax the output
        output = torch.softmax(output, dim=1)
         # Multiply the output with the transformation matrix
        output = torch.mm(output, T)
         # Calculate the loss for the training set
        sign = False
        loss_train = weighted_cross_entropy(output[idx_train], pseudo_labels[idx_train], bald[idx_train], args.beta, nclass, sign)
        # loss_train = criterion(output[idx_train], pseudo_labels[idx_train])
        # Calculate the accuracy for the training set
        acc_train = accuracy(output[idx_train], pseudo_labels[idx_train])
         # Backpropagate the loss
        loss_train.backward()
         # Update the weights
        optimizer.step()
         # Set the model to evaluation mode
        with torch.no_grad():
            model.eval()
             # Get the output of the model
            output = model(features, adj)
             # Calculate the loss for the validation set
            loss_val = criterion(output[idx_val], labels[idx_val])
             # Calculate the loss for the test set
            loss_test = criterion(output[idx_test], labels[idx_test])
             # Calculate the accuracy for the validation set
            acc_val = accuracy(output[idx_val], labels[idx_val])
             # Calculate the accuracy for the test set
            acc_test = accuracy(output[idx_test], labels[idx_test])
         # Check if the loss for the validation set is the best
        if loss_val < best:
            # Save the model to the given path
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
             # Update the best loss
            best = loss_val
             # Reset the bad counter
            bad_counter = 0
             # Store the best output
            best_output = output
        else:
            # Increment the bad counter
            bad_counter += 1
         # Check if the bad counter has reached the patience
        if bad_counter == args.patience:
            break

        # print(f'epoch: {epoch}',
        #       f'loss_train: {loss_train.item():.4f}',
        #       f'acc_train: {acc_train:.4f}',
        #       f'loss_val: {loss_val.item():.4f}',
        #       f'acc_val: {acc_val:.4f}',
        #       f'loss_test: {loss_test.item():4f}',
        #       f'acc_test: {acc_test:.4f}')
    return best_output


@torch.no_grad()
def test(adj, features, labels, idx_test, nclass, model_path, g):
    nfeat = features.shape[1]
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(f"Test set results",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}")

    return acc_test, loss_test


def main(dataset, model_path):
    # Load data from dataset
    g, adj, features, labels, idx_train, idx_val, idx_test, oadj = load_data(dataset, args.labelrate)
    # Move data to device
    g = g.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    train_index = torch.where(idx_train)[0]
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    idx_pseudo = torch.zeros_like(idx_train)
    # Get number of nodes and classes
    n_node = labels.size()[0]
    nclass = labels.max().item() + 1
    # Get the mc_adj if drop_method is dropedge
    if args.drop_method == 'dropedge':
        mc_adj = get_mc_adj(oadj, device, args.droprate) #Eq 8

    if args.labelrate != 20:
        idx_train[train_index] = True
        idx_train = generate_trainmask(idx_train, idx_val, idx_test, n_node, nclass, labels, args.labelrate)

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)
    # Initialize bald and T
    bald = torch.ones(n_node).to(device)
    # T is diagonal matrix
    T = nn.Parameter(torch.eye(nclass, nclass).to(device)) # transition matrix
    T.requires_grad = False
    # Generate random seed
    seed = np.random.randint(0, 10000)
    # Loop through stages
    for s in range(args.stage):
        #step 1 train student with T
        #step 6 Eq 13 train student without T
        #step 8,9 retrain student with T and instead of teacher
        best_output = train(model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, seed)

        #step 7 Eq 15 Update T
        T = update_T(best_output, idx_train, labels, T, device)

        #step 3  Get indices of unlabeled nodes
        idx_unlabeled = ~(idx_train | idx_test | idx_val)

        #step 5 Eq12 calculate B based on drop_method
        if args.drop_method == 'dropout':
            bald = uncertainty_dropout(adj, features, nclass, model_path, args, device)
        elif args.drop_method == 'dropedge':
            bald = uncertainty_dropedge(mc_adj, adj, features, nclass, model_path, args, device)

        #step 4 generate pseudo labels
        state_dict = torch.load(model_path)
        model = get_models(args, features.shape[1], nclass, g=g)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        # Get best output
        best_output = model(features, adj)
        # Regenerate pseudo_labels
        idx_train_ag, pseudo_labels, idx_pseudo = regenerate_pseudo_label(best_output, labels, idx_train, idx_unlabeled,
                                                                          args.threshold, device)
        # Testing
        acc_test, loss_test = test(adj, features, labels, idx_test, nclass, model_path, g)

        # plot_data_distribution(best_output.detach().cpu(), bald, labels, idx_unlabeled, idx_train, idx_test, args.dataset)
        # plot_un_conf(best_output.detach().cpu(), labels.cpu(), idx_unlabeled, bald, args.dataset)
        # plot_dis_pseudo(dataset, best_output.detach(), idx_train, idx_test, labels, idx_unlabeled, idx_pseudo, bald, s)


    return



if __name__ == '__main__':
    model_path = './save_model/%s-%s-%d-%f-%f-%f-%s.pth' % (
                    args.model, args.dataset, args.labelrate, args.threshold, args.beta, args.droprate, args.drop_method)
    main(args.dataset, model_path)


