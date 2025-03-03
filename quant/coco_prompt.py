from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json
import random
from random import shuffle
from PIL import Image

def get_prompts(json_file='/dataset/coco2014/annotations/captions_val2014.json'):
    
    list_prompts = []
    data=json.load(open(json_file,'r'))

    for ann in data['annotations']:
        list_prompts.append(ann['caption'])

    shuffle(list_prompts)
    return list_prompts

def center_resize_image(path_image, out_path, size):
    num = 0
    for filename in os.listdir(path_image):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.JPEG'):
            file_path = os.path.join(path_image, filename)
            img = Image.open(file_path)
            if filename.endswith('.JPEG') and img.mode=='RGBA':
                continue
            width, height = img.size
            square = min(width, height)
            center_x = int(width)/2
            center_y = int(height)/2
            x1 = int((width - square)/2)
            y1 = int((height - square)/2)
            box = (x1, y1, x1+square, y1+square)
            img = img.crop(box)
            image=img.resize(size, resample=Image.BICUBIC)#, box=box

            out_image = os.path.join(out_path, filename)
            image.save(out_image)
            num =  num + 1
            if num % 5000 == 0:
                print(num)

if __name__ == "__main__":
    path_image = "/dataset/imagenet/train/"
    out_path = "/dataset/imagenet/train_new/"
    if os.path.exists(path_image) and os.path.exists(out_path): 
        print("dir") 
    else: 
        print("no dir")
    center_resize_image(path_image, out_path, (256, 256))