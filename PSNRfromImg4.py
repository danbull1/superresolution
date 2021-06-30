from PIL import Image 
import os
import numpy as np
import math
import csv
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr 
import tensorflow as tf
import glob
import sys
sys.path.append(r'C:\DeepLearning\COMPX594\Data\Results2\lpips-tensorflow')
import lpips_tf
from shutil import copyfile



imagesetdir = r'C:\DeepLearning\COMPX594\Data\DS_output_SSIM2'
imagedif = r'C:\DeepLearning\COMPX594\Data\IM_test'
bc = r'C:\DeepLearning\COMPX594\Data\Results5'

outdir = r'C:\DeepLearning\COMPX594\Data\DS_output_SSIM2' + '\\imgno3.csv' ##r'C:\DeepLearning\COMPX594\Results\6Im_3hr\Im6hr_results.csv'
resize = (512,512)

dict = {}

def getBCImage(imgno):
    folder = imagedif + '\\imgset' +  imgno
    for file in os.listdir(folder):
        if file[:3] == 'ALL':
            #raw = Image.open( folder + '\\' + file)
            cloud = Image.open( folder + '\\' + 'cloud_' + file[4:])
            cloudsum = np.sum(cloud, axis=(0, 1))
            if cloudsum ==16384:
                im = folder + '\\' + file
    copyfile(im, r'C:\DeepLearning\COMPX594\Data\Results3_GAN2\raw' + imgno + '.png')
    return im

with tf.Session() as session:
    i=0
    for outputimage in os.listdir(imagesetdir):
        ##get image number as not all imagesets have an output
        
        if outputimage[:3] == 'rgb':
            i=i+1
            if i>60:
                
                imgno = outputimage[3:7]
                BCimg = 'rawrgb' + imgno + '.png'
                GN2img = r'C:\DeepLearning\COMPX594\Data\GANDS\ganrgb' + imgno +'.png'
                if os.path.exists(bc + '\\' + BCimg) and os.path.exists(GN2img):
                    HRimg = r'C:\DeepLearning\COMPX594\Data\IM_test\imgset' +imgno +'\HR_ALL_' + imgno + '.png'
                    #GNimg = 'rawganrgb' + imgno +'.png'
                    
                    
                    HR_npx = np.array(Image.open(HRimg))
                    HR_np = HR_npx[:,:,:3]
                    DS_np = np.array(Image.open(imagesetdir + '\\' + outputimage))
                    #GAN_np = np.array(Image.open(imagesetdir + '\\' + GNimg))
                    BC_npx = np.array(Image.open(bc + '\\' + BCimg))#.resize((512,512),Image.BICUBIC))
                    BC_np = BC_npx[:,:,:3]
                    GAN2_np = np.array(Image.open(GN2img))
                    #psnr_GAN = compare_psnr(HR_np, GAN_np, data_range=256)
                    #ssim_GAN = compare_ssim(HR_np, GAN_np, multichannel=True, data_range=256, gaussian_weights=True, sigma=1.5)
                    psnr_GAN2 = compare_psnr(HR_np, GAN2_np, data_range=256)
                    ssim_GAN2 = compare_ssim(HR_np, GAN2_np, multichannel=True, data_range=256, gaussian_weights=True, sigma=1.5)
                    psnr_DS = compare_psnr(HR_np, DS_np, data_range=256)
                    ssim_DS = compare_ssim(HR_np, DS_np, multichannel=True, data_range=256, gaussian_weights=True, sigma=1.5)
                    psnr_BC = compare_psnr(HR_np, BC_np, data_range=256)
                    ssim_BC = compare_ssim(HR_np, BC_np, multichannel=True, data_range=256, gaussian_weights=True, sigma=1.5)

                    image0_ph = tf.placeholder(tf.float32)
                    image1_ph = tf.placeholder(tf.float32)

                    distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='vgg')

                    lpips_BC = session.run(distance_t, feed_dict={image0_ph: HR_np, image1_ph: BC_np})
                    lpips_DS = session.run(distance_t, feed_dict={image0_ph: HR_np, image1_ph: DS_np})
                    #lpips_GAN = session.run(distance_t, feed_dict={image0_ph: HR_np, image1_ph: GAN_np})
                    lpips_GAN2 = session.run(distance_t, feed_dict={image0_ph: HR_np, image1_ph: GAN2_np})
                    lpips_BC = round(lpips_BC, 3)
                    #ms_ssim_bcx = tf.image.ssim_multiscale(HR_np, BC_np,  max_val=1.0, power_factors=(0.0448, 0.2856, 0.3001))
                    #ms_ssim_BC = ms_ssim_test.eval()

                    dict[imgno] = [imgno, psnr_BC, ssim_BC, lpips_BC, psnr_DS, ssim_DS, lpips_DS, psnr_GAN2, ssim_GAN2, lpips_GAN2]#
                    if i>90:
                        break
      

with open(outdir, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, delimiter=',')
    writer.writerow(["imgno","psnr","ssim","lpips","psnr","ssim","lpips","psnr","ssim","lpips"])
    for dictitem in dict:
        writer.writerow(dict[dictitem])



