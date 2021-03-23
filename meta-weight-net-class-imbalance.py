import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from data_utils import *
from resnet import *
from Update import *
from options import args_parser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sampling import *
from torch.utils.data import DataLoader, Dataset

args =args_parser()
print(args)

kwargs = {'num_workers': 1, 'pin_memory': True}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")



# log
current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
TAG = 'exp/{}_{}_{}_{}'.format(args.dataset, args.epochs,args.imb_factor,  current_time)
# TAG = f'alpha_{alpha}/data_distribution'
logdir = f'runs/{TAG}' if not args.debug else f'runs2/{TAG}'
writer = SummaryWriter(logdir)


# train_data_meta,train_data,test_dataset = build_dataset(args.dataset,args.num_meta)
#
# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=args.batch_size, shuffle=True, **kwargs)


# make imbalanced data
torch.manual_seed(args.seed)
# classe_labels = range(args.num_classes)
#
# data_list = {}
#
# for j in range(args.num_classes):
#     data_list[j] = [i for i, label in enumerate(train_loader.dataset.targets) if label == j]
#
# img_num_list = get_img_num_per_cls(args.dataset,args.imb_factor,args.num_meta*args.num_classes)
# print(img_num_list)
# print(sum(img_num_list))
# im_data = {}
# idx_to_del = []
# for cls_idx, img_id_list in data_list.items():
#     random.shuffle(img_id_list)
#     img_num = img_num_list[int(cls_idx)]
#     im_data[cls_idx] = img_id_list[img_num:]
#     idx_to_del.extend(img_id_list[img_num:])
#
# print(len(idx_to_del))
#
# imbalanced_train_dataset = copy.deepcopy(train_data)
# imbalanced_train_dataset.targets = np.delete(train_loader.dataset.targets, idx_to_del, axis=0)
# imbalanced_train_dataset.data = np.delete(train_loader.dataset.data, idx_to_del, axis=0)
# user_dataset=get_train_data(imbalanced_train_dataset,args)
# imbalanced_train_loader = torch.utils.data.DataLoader(
#     imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
#
#
# validation_loader = torch.utils.data.DataLoader(
#     train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
#
# best_prec1 = 0

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)

train_groups, idx_to_meta = get_train_data(train_dataset, args)
trainloader = DataLoader(DatasetSplit(train_dataset, train_groups), batch_size=100, shuffle=True)

validloader = DataLoader(DatasetSplit(train_dataset, idx_to_meta), batch_size=100, shuffle=True)
# dict_users = Divide_groups(train_groups, args.num_users)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

def main():
    global args, best_prec1
    args = args_parser()

    # create model
    model = build_model()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    vnet = VNet(1, 100, 1).cuda()
    optimizer_c = torch.optim.SGD(vnet.params(), 1e-5,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    v_record=[]



    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(args.epochs):
        v_record.append(vnet.state_dict())
        adjust_learning_rate(optimizer_a, epoch + 1)


        w,theta=train(trainloader, validloader,model,vnet, optimizer_a,optimizer_c,epoch)

        test_acc, test_loss = test_img(model, test_loader, args)
        writer.add_scalar('test_loss', test_loss, epoch + 1)
        writer.add_scalar('test_acc', test_acc, epoch + 1)

    test_vnet(v_record,args)
    writer.close()

        # evaluate on validation set
    #     prec1 = validate(test_loader, model, criterion, epoch)
    #
    #     # remember best prec@1 and save checkpoint
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)
    #
    # print('Best accuracy: ', best_prec1)







if __name__ == '__main__':
    main()




