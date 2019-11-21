
# native imports
import os
import glob
import random

# open source imports
import torch
import torch.nn
import cv2
import numpy as np

# custom imports
from options import args
from model import MyNet
use_cuda = torch.cuda.is_available()

##############################################################################################
##############################################################################################
##############################################################################################

def readfile(filename):
    im = cv2.imread(filename)
    data = torch.from_numpy(im).float().permute(2,0,1) / 255.0
    data = data.cuda().unsqueeze(0)
    return data

def visualize(output,viewtime=0):
    _,target = torch.max(output,1)
    im_target = target.data.cpu().numpy()
    n_labels = len(np.unique(im_target))
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
    cv2.imshow('output',im_target_rgb)
    cv2.waitKey(viewtime)

def save(output,fname,out_dir=args.out_dir):
    out_path = os.path.join(dir,fname)

# define model
loadedparams = torch.load(args.model_dir,map_location=args.device)
model = MyNet( 3 )
if use_cuda:
    model.cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))
model.eval()

if args.model_dir != '':
    model.load_state_dict(loadedparams['model'],strict=True)
else:
    print('PLEASE LOAD A MODEL_DIR AS AN OPTION')

# main function for testing
if __name__ == '__main__':

    # process a single image
    if args.input != '':
        data = readfile(args.input)
        with torch.no_grad():
            output = model(data)[0]

        print(output)
        quit()
        # visualize predicted output
        visualize(output)

    # process an entire directory and save it
    elif args.input_dir != '':
        FILES = glob.glob(os.path.join(args.input_dir,'*'))
        for f in FILES:
            data = readfile(f)
            with torch.no_grad():
                output = model(data)[0]

            # visualize predicted output
            visualize(output)




