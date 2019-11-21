import argparse

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')


# model hyper parameters
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                    help='number of superpixels')
parser.add_argument('--nChannel', metavar='N', default=1000, type=int,
                    help='number of channels')

# training code options
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--compactness', metavar='C', default=100, type=float,
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--training_dir', metavar='FILENAME',default='../data/DIV2K_train_LR_bicubic/x4',type=str,
                    help='input image file name', required=False)
parser.add_argument('--device', metavar='mode',default='cuda',type=str,
                    help='specify cuda or cpu', required=False)

# testing code options
parser.add_argument('--model_dir', metavar='FILENAME',default='',type=str,
                    help='path to model', required=False)
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=False)
parser.add_argument('--input_dir', metavar='FILENAME',
                    help='input directory to process during testing', required=False)
parser.add_argument('--out_dir', metavar='FILENAME',default='out',help='output directory to save files to', required=False)
args = parser.parse_args()
