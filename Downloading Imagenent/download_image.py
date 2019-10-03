''' This script downloads the images of the given class '''
import json
from urllib.request import urlretrieve
from tqdm import *
import ipdb
import argparse 
import os, sys
from random import shuffle
import random
import json

def random_classes(num_classes):
    num_classes=int(num_classes)
    f = open("imagenet_class_index.json","r")
    class_index = json.loads(f.read())
    class_to_wnid = {}
    for i in range(1000):
        wnid, class_name = class_index[str(i)]
        class_to_wnid[class_name.lower()] = wnid
    
    all_classes = class_to_wnid.keys()
    if type(all_classes)!='list':
        list_classes_all=list(all_classes)
    else:
        list_classes_all=all_classes
    random.shuffle(list_classes_all)
    list_classes_all=list_classes_all[:num_classes]
    return list_classes_all

parser = argparse.ArgumentParser(description="Download images to given folder")
parser.add_argument("-c", type = str, help = "The class whose images will be downloaded")
parser.add_argument("-n", type = str, default = "all", help = "The number of images to be downloaded (with auto upper limit)")
parser.add_argument("-o", type = str, help = "output folder for downloaded images")
args = parser.parse_args()

# load url file 
with open('fall11_urls.txt', 'r',encoding = "ISO-8859-1") as f:
    urls = f.readlines() 
    urls = [str(url.strip()) for url in urls]
    

# load class to wnid 
with open("imagenet_class_index.json","r") as f:
    class_index = json.loads(f.read())
class_to_wnid = {}
for i in range(1000):
    wnid, class_name = class_index[str(i)]
    class_to_wnid[class_name.lower()] = wnid

# Split urls and image ids
classes=args.c
x=0
if classes.startswith('random'):
    classes=random_classes(num_classes=classes.split("_")[1])
    x=1
if x==1:
    my_list=classes
else:
    my_list = classes.split(",")
for x in my_list:
    class_name = x
    class_wnid = str(class_to_wnid[class_name])
    urls_wnid  = [url.split("\t")[-1] for url in urls if class_wnid in url]
    imageids   = [url.split("\t")[0] for url in urls if class_wnid in url]
    assert len(urls_wnid) == len(imageids), "urls_wnid and imageids do not have the same length"
    print("[LOG]: Classname: {}".format(class_name))
    print("[LOG]: WordNet ID: {}".format(class_wnid))


    # Number of images to download 
    num_images = 0
    if args.n == "all":
        num_images = len(imageids)
    elif int(args.n) > len(imageids):
        print("[WARN]: User argument 'number of images' exceeds max images in dataset, Auto capping to MAX")
        num_images = len(imageids)
    else:
        num_images = int(args.n)
    print("[LOG]: Downloading {} images out of a total of {}".format(num_images, len(imageids)))

    # ipdb.set_trace()
    # Shuffle  
    shuff_idxs = list(range(len(imageids)))
    shuffle(shuff_idxs)

    # instead of reading num_images we read num=2*num_images so that end up with num_images no. of images
    if num_images*2<len(imageids):    
        num=2*num_images
    else:
        num=len(imageids)

    urls_wnid = [urls_wnid[i] for i in shuff_idxs][:num]
    imageids =  [imageids[i] for i in shuff_idxs][:num]

    # Download folder 
    dest=args.o
    if dest==None:
        dest='ILSVRC'
    output_folder = os.path.join(dest,x)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    assert os.path.isdir(output_folder), "Invalid output folder path provided by user"
    print("[LOG]: output folder: {}".format(output_folder))

    # Downloading images from url
    for url, imid in tqdm(zip(urls_wnid, imageids)):

        fname = imid 
        if url.strip().split(".")[-1].lower() in ['jpg', 'gif', 'jpeg','png', 'bmp', 'thb', 'jpe', 'tif', 'pjpeg']:
            fname = fname + "." + url.strip().split(".")[-1] # keep the file extension in fname 
        else: 
            fname = fname + ".jpg" # default use .jpg file extension
        num_images-=1

        try:
            urlretrieve(url, filename=os.path.join(output_folder, fname))
        except:
            num_images+=1
            print("Could not retrieve url: ", url, " | skipping") 
        if num_images==0:
            break




