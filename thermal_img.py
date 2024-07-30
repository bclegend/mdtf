import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# define the normalize function for the raw picture
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


# set the file path
targ_dir = './rawdata/all/'
save_dir = './flirlepton/all/'

# make save directory if not exist 
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


for class_name in os.listdir(targ_dir):
    print(class_name)
    class_dir = targ_dir + class_name + '/'

    # make the classis directory if not exists
    save_class_dir = save_dir + class_name + '/'
    if not os.path.isdir(save_class_dir):
        os.mkdir(save_class_dir)

    for picture in os.listdir(class_dir):
        pic_dir = class_dir + picture
        save_pic_dir = save_class_dir + picture
        print(pic_dir)
        img_raw = np.asarray(Image.open(pic_dir), dtype=np.uint16)
        img = normalize(img_raw)
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Hide axes
        plt.savefig(save_pic_dir, bbox_inches='tight', pad_inches=0)
