##Updates an image colours
##creates a lookup dictionary between DS images and HR images. 
##This dictionary is then used to look up the closest histogram to an image, and then apply the histogram match from skimage to that image

from PIL import Image 
import os
import numpy as np
import math
import csv
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from skimage import exposure
import skimage
import seaborn as sns

dir_list = r'C:\DeepLearning\COMPX594\Data\IM' ##HR images in imagesets
DS_output = r'C:\DeepLearning\COMPX594\Data\DS_output' ##DeepSUM output images


unprocessed_res = r'C:\DeepLearning\COMPX594\Data\DS_output_SSIM2' ##dir with images to update
out = r'C:\DeepLearning\COMPX594\Data\DS_output_PL' ##output
###bicubic = r'C:\DeepLearning\COMPX594\Data\TEST_bicubic' + '\\raw' + imgno + '.png'
##dsgan =r'C:\DeepLearning\COMPX594\Data\DS_output_SSIM2\col'
#rawgan = r'C:\DeepLearning\COMPX594\Data\TEST_GAN\Ouput_raw'

def create_dic():
    print ("creating dictionary")
    hist_dic = {}
    i=0
    for dir_name in os.listdir(dir_list):
        for filename in os.listdir(dir_list + '\\' + dir_name):
            if filename[:6] == 'HR_ALL': 
                i=i+1
                imgno = filename[7:11]
                hrimage = filename
                img = cv2.imread(dir_list + '\\' + dir_name + '\\' + filename)
                template = np.array(img)
                ds = DS_output + '\\rgb' + imgno + '.png'
                img = cv2.imread(dir_list + '\\' + dir_name + '\\' + filename)
                source= np.array(img)
                #hist_source = cv2.calcHist(source, [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
                if os.path.isfile(ds):
                    hist_dic[filename] = source, template
                if i>500:
                    break
    return hist_dic

def adjust_image(img, update_img, hist_dic, outloc):
    #r,g,b = cv2.split(img) 

    r  = cv2.calcHist([img],[0],None,[256],[0,256])
    g  = cv2.calcHist([img],[1],None,[256],[0,256])
    b  = cv2.calcHist([img],[2],None,[256],[0,256])
    #hist_image = cv2.calcHist([img],[0,1,2],None,[256,256,256],[0,256,0,256,0,256])
    results = {}
    for key, value in hist_dic.items():
        source = value[0]
        #source_channels = cv2.split(source) 
        rs  = cv2.calcHist([source],[0],None,[256],[0,256])
        gs  = cv2.calcHist([source],[1],None,[256],[0,256])
        bs  = cv2.calcHist([source],[2],None,[256],[0,256])

        #hist_source = cv2.calcHist(source, [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
        result_r = cv2.compareHist(r, rs, cv2.HISTCMP_CORREL)
        result_b = cv2.compareHist(g, gs, cv2.HISTCMP_CORREL)
        result_g = cv2.compareHist(b, bs, cv2.HISTCMP_CORREL)
        results[key] = result_r + result_g + result_b
    max_key = max(results, key=results.get)
    template = hist_dic[max_key][1]
    #matched = hist_match(image, template)
    matched = exposure.match_histograms(update_img, template, multichannel=True)
    cv2.imwrite(outloc, matched)


#image = r'C:\DeepLearning\COMPX594\Data\Results3\rgb0099.png'
#img = cv2.imread(image)
#hist_image = cv2.calcHist([img],[0,1,2],None,[256,256,256],[0,256,0,256,0,256])
hist_dic = create_dic()

print ("done dictionary")


for rgb in os.listdir(unprocessed_res):
    if rgb[:3] == 'rgb':
        imgno = rgb[3:7]
        img = cv2.imread(unprocessed_res + '\\' + rgb)
        update_img = cv2.imread(r'C:\DeepLearning\COMPX594\Data\GANDS\ganOuput_rgb' + imgno + '.png')
        if os.path.exists(r'C:\DeepLearning\COMPX594\Data\GANDS\ganOuput_rgb' + imgno + '.png'):
            adjust_image(img, update_img, hist_dic, r'C:\DeepLearning\COMPX594\Data\GANDS\gan' + rgb)

