import torch
import argparse

#----CommandLineArguements------------
parser = argparse.ArgumentParser(description='PyTorch Depth Map Prediction')

parser.add_argument('--epochs', type=int, default = 10, metavar='N',
                    help='number of epochs to train(default: 20) ')

parser.add_argument('--batch_size','--b',type=int,default=4,metavar="batch",
                    help='input batch size for training(default: 32) ')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of Data loading workers (default: 16)')

parser.add_argument("--lr",'--learning-rate',type=float,default=1e-4,metavar='learning rate',
                    help="initial learning rate (default 1e-4)")

parser.add_argument("--weight_decay",'--wd',type=float,default=1e-4,metavar="wd",
                    help="weight decay(default: 1e-4)")

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--eval', type=str, default='',
                    help='evaluate models on validation set')

args = parser.parse_args()


#----GlobalVariables------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"