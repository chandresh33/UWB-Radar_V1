import numpy as np   # to handle matrix and data operation
import pandas as pd  # to read csv and handle dataframe
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, AutoMinorLocator
from tqdm import tqdm
import copy
from mpl_toolkits.mplot3d import Axes3D
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

import torchvision.transforms as transforms
import torchvision.datasets

# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import scipy.stats as sps
# import scipy.signal as spy_sig
# import scipy.ndimage as ndimage
# import scipy.optimize as spy_opt
# import random
# import os
# import pickle

random_seed = 1
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = xm.xla_device()

# torch_xla.core.xla_model.set_rng_state(1, device=device)
np.random.seed(0)
torch.manual_seed(0)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(num_classes=10)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  ## googlenet
        # self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

        self.loss = nn.CrossEntropyLoss()

    @auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def load_dataset(download=False, dataset='cifar'):
    global train_dataset_
    dataset = dataset.lower()
    batch_size = 100

    trans_og = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # trans_og = transforms.Compose([transforms.Resize(96),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    AlexTransform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    imagenet_trans = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trans =  transforms.Compose(
    #   [transforms.ToTensor(),
    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trans = AlexTransform

    if dataset == 'cifar':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        print("CIFAR10")
        train_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", train=True,
                                                      transform=trans, download=download)
        test_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", train=False, transform=trans)

    elif dataset == 'mnist':
        trans_og = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        print("MNIST")
        train_dataset_ = torchvision.datasets.MNIST(root=r"Dataset/", train=True,
                                                    transform=trans_og, download=download)
        test_dataset_ = torchvision.datasets.MNIST(root=r"Dataset/", train=False, transform=trans_og)

    elif dataset == 'imagenet':
        print("ImageNet")
        trans = imagenet_trans
        train_dataset = torchvision.datasets.ImageNet(root=r"Dataset/", split='train')
        test_dataset_ = torchvision.datasets.ImageNet(root=r"Dataset/", split='test', target_transform=trans)

    else:
        trans = transforms.Compose([transforms.Resize(96),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10",
                                                      transform=trans, download=download)
        test_dataset_ = torchvision.datasets.CIFAR10(root=r"Dataset/CIFAR10", train=False, transform=trans)

    train_loader_ = DataLoader(dataset=train_dataset_, num_workers=2, batch_size=batch_size, shuffle=True)
    test_loader_ = DataLoader(dataset=test_dataset_, batch_size=100, shuffle=True)

    return train_loader_, test_loader_


def model_tester(model_, test_set):
    # Testing Accuracy
    model_.model.eval()
    model_.model.to(device)
    criterion = nn.CrossEntropyLoss()

    correct, total, running_loss = 0, 0, 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_.model(images)

            test_loss = criterion(outputs, labels).to(device)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += test_loss.item() * images.size(0)

    accuracy = correct / total
    epoch_loss = running_loss / len(test_set)

    return accuracy, epoch_loss


def get_test_names(ex_model_og):
    ex_model = copy.deepcopy(ex_model_og)
    # params, names = count_parameters(ex_model.model)

    model_obj = copy.deepcopy(ex_model.model.state_dict())
    new_names = []

    for p_name, module in list(ex_model.model.named_modules())[1:]:
        nn_ = p_name + ".weight"
        if isinstance(module, nn.Conv2d):
            new_names.append(nn_)
        elif isinstance(module, nn.Linear):
            new_names.append(nn_)

    return new_names, model_obj


def test_harness(model_, test_set):
    mod_names, mod_obj = get_test_names(model_)
    res_dict = {}
    acc, loss = model_tester(model_, test_set)
    res_dict['OG'] = (acc, loss)
    print("OG", (acc, loss))
    print("\n")

    for name in tqdm(mod_names):
        model_clone = copy.deepcopy(model_)
        new_obj = copy.deepcopy(mod_obj)

        weight = copy.deepcopy(mod_obj[name])
        filtered_weight = torch.ones_like(weight)
        new_obj[name] = filtered_weight
        model_clone.model.load_state_dict(new_obj)
        acc, loss = model_tester(model_clone, test_set)
        res_dict[name] = (acc, loss)

        print(name, (acc, loss))
        print("\n")

        del filtered_weight

    return res_dict


train_loader, test_loader = load_dataset(download=False, dataset='mnist')
# WRAPPED_MODEL = xmp.MpModelWrapper(Model())
dataset_name = "MNIST"
e_no = 10
path = 'Saved_models/'
file_name = 'Init_3/ResNet18_' + dataset_name + "_" + str(e_no) + 'epochs.ckpt'
checkpoint = torch.load(path+file_name, map_location=torch.device("cpu"))

model = Model()
model.load_state_dict(checkpoint)
# red_data = test_harness(model, test_loader)
weight = model.model.layer4[1].conv1.weight.detach()

print(weight[0][0])
weight = weight/weight
print(weight[0][0])

