import argparse
import glob
from scipy import misc
from utils import dataAugmentation,createGaussianLabel
import numpy as np
import os
import cv2
from tqdm import tqdm

def get_parser():
    
    parser = argparse.ArgumentParser('preprocess')
    parser.add_argument('--inputPath', '-i', required=True)
    parser.add_argument('--outputPath', '-o', required=True)
    parser.add_argument('--outputsize','-s', type=sizes,default=(256,256,3))
    parser.add_argument("--augmentation", '-a',  type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument("--GaussianSize", '-g',  type=int, default=9)
    return parser


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sizes(s):
    try:
        x, y, c = map(int, s.split(','))
        return (x, y, c)
    except:
        raise argparse.ArgumentTypeError("size must be x,y,c")

def preprocess(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    jpgFiles = glob.glob(args.inputPath+'/*.jpg')
    os.makedirs(args.outputPath, exist_ok=True)
    for f in  tqdm(jpgFiles):
        image=cv2.imread(f)
        img=cv2.resize(image, args.outputsize[:2], cv2.INTER_CUBIC)
        label=createGaussianLabel(f.replace(".jpg",".json"),args.outputsize,image.shape,args.GaussianSize)
        if args.augmentation:
            images,labels=dataAugmentation([img],[label])
            for i in range(len(images)):
                basename = os.path.basename(f)
                new_name = os.path.join(args.outputPath, basename).replace(".jpg", ".png")
                cv2.imwrite(new_name, images[i])
                np.save(new_name.replace('.png','.npy'), labels[i].astype(np.uint8))
        else:
            basename = os.path.basename(f)
            new_name = os.path.join(args.outputPath, basename).replace(".jpg", ".png")
            cv2.imwrite(new_name,img) 
            np.save(new_name.replace('.png','.npy'),label)

if __name__ == "__main__":
   preprocess()