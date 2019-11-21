#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import segmentation
import torch.nn.init

import glob
import os
import random
from itertools import count

from options import args
from model import MyNet

use_cuda = torch.cuda.is_available()
###############################################################################################
###############################################################################################

def loadData():
    files = glob.glob(os.path.join(args.training_dir,"*"))
    f = random.sample(files,1)[0]
    im = cv2.imread(f)
    im = crop(im)
    data = torch.from_numpy(im).float().permute(2,0,1) / 255.0
    data = data.cuda().unsqueeze(0)
    return im, Variable(data)

def crop(img,width=128,height=128):
    h,w,_ = img.shape
    xpos = random.randint(0,w-width-1)
    ypos = random.randint(0,h-height-1)
    return img[ypos:ypos+height,xpos:xpos+width]

def save(model,k):
    data = {'model': model.state_dict()}
    data['k'] = k
    torch.save(data,'segnet.pth')

# define model
model = MyNet( 3 )
if use_cuda:
    model.cuda()
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))

#for batch_idx in range(args.maxIter):
for batch_idx in count():
# load image
    im,data = loadData()

    # slic
    labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        cv2.imshow( "output", im_target_rgb )
        cv2.waitKey(10)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[ l_inds[ i ] ]
        u_labels_per_sp = np.unique( labels_per_sp )
        hist = np.zeros( len(u_labels_per_sp) )
        for j in range(len(hist)):
            hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
        im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
    target = torch.from_numpy( im_target )
    if use_cuda:
        target = target.cuda()
    target = Variable( target )
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    #print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
    print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())

    # save after every 800 iterations
    if batch_idx % 800 == 0:
        save(model, nLabels)

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        save(model, nLabels)
        print('model saved')
        break

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )

cv2.imwrite( "output.png", im_target_rgb )

