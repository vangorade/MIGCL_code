import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from model import load_model
from modules import LogisticRegression
from data_statistics import get_data_mean_and_stdev, get_data_nclass
from transfer_datasets import DATASET
import numpy as np

parser = argparse.ArgumentParser(description='Linear Evaluation')
parser.add_argument('--model', default="orig", type=str, help="orig/RC/LBE")
parser.add_argument('--logistic_batch_size', default=128, type=int, metavar='B', help='logistic_batch_size batch size')
parser.add_argument('--logistic_epochs', default=100, type=int, help='logistic_epochs')
parser.add_argument('--batch_size', default=256, type=int, metavar='B', help='training batch size')
parser.add_argument('--workers', default=12, type=int, help='workers')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--lamb', default=0.1, type=float, help='weight of regularization term')
parser.add_argument('--zeta', default=0.01, type=float, help='variance')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--model_dir', default='output/checkpoint/', type=str, help='model save path')
parser.add_argument('--dataset', default='CIFAR10', help='[CIFAR10, CIFAR100]')
parser.add_argument('--testset', default='cu_birds', help='[CIFAR10, CIFAR100, aircraft, cu_birds, dtd, fashionmnist, mnist, traffic_sign, vgg_flower]')
args = parser.parse_args() 

def train(loader, encoder, model, criterion, optimizer):
    loss_epoch = 0
    model.train()
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        with torch.no_grad():
            h, _ = encoder(x.cuda())
        output = model(h)
        loss = criterion(output, y.cuda())
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 50 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
    return loss_epoch

def test(loader, encoder, model):
    right_num = 0
    all_num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
           
            h, _ = encoder(x)
            output = model(h)
            predicted = output.argmax(1)
            right_num += (predicted == y).sum().item()
            all_num += y.size(0)
    accuracy = right_num*100./all_num
    return accuracy

def load_transform(dataset, size=32):
    mean, std = get_data_mean_and_stdev(dataset)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    return transform

def main():
    root = "../datasets"
    if args.testset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=load_transform('CIFAR10', 32))
        test_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=load_transform('CIFAR10', 32))
    elif args.testset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=load_transform('CIFAR100', 32))
        test_dataset = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=load_transform('CIFAR100', 32))
    else:
        train_dataset = DATASET[args.testset](train=True, image_transforms=load_transform(args.testset, 32))
        test_dataset = DATASET[args.testset](train=False, image_transforms=load_transform(args.testset, 32))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers)

    log_dir = "output/log/" + args.testset + '_%s/'%args.model
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    args.model_dir = args.model_dir + args.dataset + '_%s/'%args.model
    epoch_dir = args.model_dir + suffix + '_epoch_%d.pt'%args.epochs
    print("Loading {}".format(epoch_dir))

    encoder, _, _, _ = load_model(args, get_data_nclass(args.dataset), reload_model=True, load_path=epoch_dir)
    encoder = encoder.cuda()
    encoder.eval()

    # Logistic Regression
    n_classes = get_data_nclass(args.testset)
    model = LogisticRegression(encoder.n_features, n_classes).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.
    test_log_file = open(log_dir + suffix + '_LR.txt', "w")
    for epoch in range(args.logistic_epochs):
        loss_epoch = train(train_loader, encoder, model, criterion, optimizer)
        print("Train Epoch [{}]\t Average loss: {}".format(epoch, loss_epoch/len(train_loader)))
        print("Train Epoch [{}]\t Average loss: {}".format(epoch, loss_epoch/len(train_loader)), file=test_log_file)
        test_log_file.flush()

        # final testing
        test_current_acc = test(test_loader, encoder, model)
        if test_current_acc > best_acc:
            best_acc = test_current_acc
        print("Test Epoch [{}]\t Accuracy: {}\t Best Accuracy: {}\n".format(epoch, test_current_acc, best_acc))
        print("Test Epoch [{}]\t Accuracy: {}\t Best Accuracy: {}\n".format(epoch, test_current_acc, best_acc), file=test_log_file)
        test_log_file.flush()
        scheduler.step()

    print("Final Best Accuracy: {}".format(best_acc))
    print("Final Best Accuracy: {}".format(best_acc), file=test_log_file)
    test_log_file.flush()

if __name__ == "__main__":
    main()