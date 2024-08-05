import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# check if the crop folder exist 
if not os.path.isdir('./crop/'):
        os.mkdir('./crop/')

# ## Only motor
# x = 280
# y = 150
# w = 130
# h = 120
## inner
x = 300
y = 190
w = 90
h = 60


# ## Only load
# x = 50
# y = 120
# w = 200
# h = 160

# ## WTF is that?
# x = 340
# y = 30
# w = 50
# h = 50

## crop thermal images
for cla in os.listdir('./flirlepton/'):
    if not os.path.isdir('./crop/' + cla):
        os.mkdir('./crop/' + cla)
    for im in os.listdir('./flirlepton/' + cla):
        img = cv2.imread('./flirlepton/' + cla + '/' + im)
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite('./crop/' + cla + '/' + im, crop_img)
