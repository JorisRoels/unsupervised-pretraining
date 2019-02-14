
"""
    This is a script that trains U-Nets from scratch
    Usage:
        python train.py --method 2D
"""

"""
    Necessary libraries
"""
import os
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from copy import deepcopy

from data.datasets import StronglyLabeledVolumeDataset, UnlabeledVolumeDataset
from networks.unet import UNet2D, UNet3D
from util.io import imwrite3D
from util.losses import CrossEntropyLoss, MSELoss
from util.preprocessing import get_augmenters_2d, get_augmenters_3d
from util.validation import segment
from util.metrics import jaccard, dice, accuracy_metrics

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--method", help="Specifies 2D or 3D U-Net", type=str, default="2D")

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")
parser.add_argument("--write_dir", help="Writing directory", type=str, default=None)
parser.add_argument("--data_train", help="Data for training", type=str, default="../../data/epfl/training.tif")
parser.add_argument("--labels_train", help="Labels for training", type=str, default="../../data/epfl/training_groundtruth.tif")
parser.add_argument("--data_test", help="Data for testing", type=str, default="../../data/epfl/testing.tif")
parser.add_argument("--labels_test", help="Labels for testing", type=str, default="../../data/epfl/testing_groundtruth.tif")
parser.add_argument("--data_unsupervised_train", help="Data for unsupervised training", type=str, default=None)
parser.add_argument("--data_unsupervised_test", help="Data for unsupervised testing", type=str, default=None)
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=100)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="128,128")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=64)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)", type=int, default=4)
parser.add_argument("--group_norm", help="Use group normalization instead of batch normalization", type=int, default=0)
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=1)
parser.add_argument("--class_weight", help="Percentage of the reference class", type=float, default=(0.5))

# optimization parameters
parser.add_argument("--preprocess", help="Type of preprocessing", type=str, default="z") # z or unit
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=100)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=4)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
weight = torch.FloatTensor([1-args.class_weight, args.class_weight]).cuda()
loss_fn_seg = CrossEntropyLoss(weight=weight)
loss_fn_rec = MSELoss()

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    os.mkdir(os.path.join(args.write_dir, 'segmentation_last_checkpoint'))
    os.mkdir(os.path.join(args.write_dir, 'segmentation_best_checkpoint'))

"""
    Load the data
"""
if args.method == "2D":
    input_shape = (1, args.input_size[0], args.input_size[1])
else:
    input_shape = args.input_size
print('[%s] Loading data' % (datetime.datetime.now()))
# augmenters
if args.method == "2D":
    train_xtransform_us, train_ytransform_us, test_xtransform_us, test_ytransform_us = get_augmenters_2d(augment_noise=(args.augment_noise==1))
    train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters_2d(augment_noise=(args.augment_noise==1))
else:
    train_xtransform_us, train_ytransform_us, test_xtransform_us, test_ytransform_us = get_augmenters_3d(augment_noise=(args.augment_noise==1))
    train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters_3d(augment_noise=(args.augment_noise==1))
# load data
train = StronglyLabeledVolumeDataset(args.data_train, args.labels_train, input_shape, transform=train_xtransform, target_transform=train_ytransform, preprocess=args.preprocess)
test = StronglyLabeledVolumeDataset(args.data_test, args.labels_test, input_shape, transform=test_xtransform, target_transform=test_ytransform, preprocess=args.preprocess)
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.test_batch_size)
if args.data_unsupervised_train is not None:
    train_unsupervised = UnlabeledVolumeDataset(args.data_unsupervised_train, input_shape, transform=train_xtransform_us)
    test_unsupervised = UnlabeledVolumeDataset(args.data_unsupervised_test, input_shape=input_shape, transform=train_xtransform_us)
    train_loader_unsupervised = DataLoader(train_unsupervised, batch_size=args.train_batch_size)
    test_loader_unsupervised = DataLoader(test_unsupervised, batch_size=args.test_batch_size)
else:
    train_loader_unsupervised = None
    test_loader_unsupervised = None

"""
    Setup optimization for unsupervised training if necessary
"""
if args.data_unsupervised_train is not None:
    print('[%s] Setting up optimization for unsupervised training if necessary' % (datetime.datetime.now()))
    if args.method == "2D":
        net_us = UNet2D(out_channels=1, feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm == 1), pretrain_unsupervised=(args.data_unsupervised_train is not None))
    else:
        net_us = UNet3D(out_channels=1, feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1), pretrain_unsupervised=(args.data_unsupervised_train is not None))

    """
        Train the network unsupervised
    """
    print('[%s] Training network unsupervised' % (datetime.datetime.now()))
    net_us.train_net(train_loader=train_loader_unsupervised, test_loader=test_loader_unsupervised,
                  loss_fn=loss_fn_rec, lr=args.lr, step_size=args.step_size, gamma=args.gamma,
                  epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
                  log_dir=args.log_dir)

"""
    Setup optimization for supervised training
"""
print('[%s] Setting up optimization for supervised training' % (datetime.datetime.now()))
if args.method == "2D":
    net = UNet2D(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm == 1))
else:
    net = UNet3D(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
if args.data_unsupervised_train is not None:
    net.encoder = deepcopy(net_us.encoder)

"""
    Train the network supervised
"""
print('[%s] Training network supervised' % (datetime.datetime.now()))
net.train_net(train_loader=train_loader, test_loader=test_loader,
              loss_fn=loss_fn_seg, lr=args.lr, step_size=args.step_size, gamma=args.gamma,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=args.log_dir)

"""
    Validate the trained network
"""
print('[%s] Validating the trained network' % (datetime.datetime.now()))
test_data = test.data
test_labels = test.labels
segmentation_last_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
a, p, r, f = accuracy_metrics(segmentation_last_checkpoint, test_labels)
print('[%s] Results last checkpoint:' % (datetime.datetime.now()))
print('[%s]     Jaccard: %f' % (datetime.datetime.now(), j))
print('[%s]     Dice: %f' % (datetime.datetime.now(), d))
print('[%s]     Accuracy: %f' % (datetime.datetime.now(), a))
print('[%s]     Precision: %f' % (datetime.datetime.now(), p))
print('[%s]     Recall: %f' % (datetime.datetime.now(), r))
print('[%s]     F-score: %f' % (datetime.datetime.now(), f))
net = torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch'))
segmentation_best_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
a, p, r, f = accuracy_metrics(segmentation_best_checkpoint, test_labels)
print('[%s] Results best checkpoint:' % (datetime.datetime.now()))
print('[%s]     Jaccard: %f' % (datetime.datetime.now(), j))
print('[%s]     Dice: %f' % (datetime.datetime.now(), d))
print('[%s]     Accuracy: %f' % (datetime.datetime.now(), a))
print('[%s]     Precision: %f' % (datetime.datetime.now(), p))
print('[%s]     Recall: %f' % (datetime.datetime.now(), r))
print('[%s]     F-score: %f' % (datetime.datetime.now(), f))

"""
    Write out the results
"""
if args.write_dir is not None:
    print('[%s] Writing the output' % (datetime.datetime.now()))
    imwrite3D(segmentation_last_checkpoint, os.path.join(args.write_dir, 'segmentation_last_checkpoint'), rescale=True)
    imwrite3D(segmentation_best_checkpoint, os.path.join(args.write_dir, 'segmentation_best_checkpoint'), rescale=True)

print('[%s] Finished!' % (datetime.datetime.now()))