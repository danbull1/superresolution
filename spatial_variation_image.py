import sys
import numpy as np
from scipy import signal
from scipy import ndimage
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statistics import mean
import random

patch_size = 2

im1=r'C:\Users\danbu\Google Drive\DeepLearning\Results\HRimages\HR61.png'
im2=r'C:\Users\danbu\Google Drive\DeepLearning\Results\VariationLoss\varloss2\images\test_61.png'
img1 = np.asarray(Image.open(im1))
img2 = np.asarray(Image.open(im2))


bush_var = []
farm_var = []
bush_pts = []
farm_pts = []
for i in range(100):
    x = random.randint(100, 300)
    y = random.randint(0, 220)
    bush_pts.append((x,y))
    hr_patch_bush = img2[x:x+patch_size,y:y+patch_size]
    hr_var_bush = hr_patch_bush.var()
    bush_var.append(hr_var_bush)

for i in range(100):
    x = random.randint(120, 250 )
    y = random.randint(250, 400)
    farm_pts.append((x,y))
    hr_patch_farm = img2[x:x+patch_size,y:y+patch_size]
    hr_var_farm = hr_patch_farm.var()
    farm_var.append(hr_var_farm)

bush_mean_var = mean(bush_var)
farm_mean_var = mean(farm_var)

print ("bush mean var : " + str(bush_mean_var))
print ("farm mean var : " + str(farm_mean_var))

fig, ax = plt.subplots()
imgplot = ax.imshow(img1)
for i in range(100):
    rect = patches.Rectangle(bush_pts[i], 1, 1, linewidth=3, edgecolor='red', fill=True, facecolor='red',alpha = 1)
    rect2 = patches.Rectangle(farm_pts[i], 1, 1, linewidth=3, edgecolor='yellow', fill=True, facecolor='red',alpha = 1)
    ax.add_patch(rect)
    ax.add_patch(rect2)

plt.show()
#plt.savefig(r'c:\temp\styleloss_plot3.png')


