#creates imageset from HR and LR aligned images + cloud
##mask raster defines AOI
##outputs a directory for each imageset with a patch for each colour


from PIL import Image 
import random
import math
import os
import numpy as np


cropwidth = 128
cropheight = 128

dir = r'C:\DeepLearning\COMPX594\Data\WW'##firectory with LR Sentinel images AND cloud images

saveloc =  r'C:\DeepLearning\COMPX594\Data\IM_test'##save out location

hrimage = r'C:\DeepLearning\COMPX594\Data\WW\HR.png'##hr raster

mask = r'C:\DeepLearning\COMPX594\Data\WW\mask.png'##raster defines area of interest i.e. 0s are not used 1s are used

cloudtemplate = r'C:\DeepLearning\COMPX594\Data\WW\fakecloud.png' ##used for fake cloud

i=0

##no of imagesets to make
maximagsets=100

def makenostr(i):
    numi = str(i)
    if len(numi) == 1:
        numi = '000' + numi
    if len(numi) == 2:
        numi = '00' + numi
    if len(numi) == 3:
        numi = '0' + numi
    return numi

Image.MAX_IMAGE_PIXELS = None
maskimg = Image.open(mask)
hrimg = Image.open(hrimage)

sentinel = {}
channel_dict ={0: 'red', 1: 'green', 2: 'blue', 3: 'NIR'}

for image in  os.listdir(dir):
    if image[-3:] == "png" and image[:2] != "HR" and image[-9:-4] != "cloud" and image[:-4] != "mask":
          imagename = image[:-4]
          sentinel[imagename] = Image.open(dir + "\\" + image)
    if image[-3:] == "png" and image[-10:-4] == "_cloud":
          imagename = image[:-4]
          sentinel[imagename] = Image.open(dir + "\\" + image)

while i<maximagsets:
    numi = makenostr(i)
    width, height = 10980-1, 10980-1
    randheight = random.randint(0, height - cropheight)
    randwidth = random.randint(0, width - cropwidth)
    print(randwidth)
    print(randheight)
    xmax = randwidth + cropwidth
    ymax = randheight + cropheight
    area = (randwidth, randheight, xmax, ymax)
    #Image.MAX_IMAGE_PIXELS = None
    #maskimg = Image.open(mask)
    topleft = maskimg.getpixel((randwidth, randheight))
    topright = maskimg.getpixel(( randwidth + cropwidth, randheight))
    bottomleft = maskimg.getpixel((randwidth, randheight + cropheight))
    bottomright = maskimg.getpixel((randwidth + cropwidth, randheight + cropheight))
    print(topleft)
    print(topright)
    print (bottomleft)
    print (bottomright)

    ##makes sure random data is within the mask
    if topleft !=1 or topright !=1 or bottomleft !=1 or bottomright !=1:
        print ("within mask")
    else:
        hrarea = (randwidth*4, randheight*4, xmax*4, ymax*4)
        newdir = saveloc + "\\imgset" + numi
        os.mkdir(newdir)
        hrcropped = hrimg.crop(hrarea)
        hrcropped.save(newdir + '\\HR_ALL_' + numi +'.png')
        channelno=0
        for channel in hrimg.split(): 
            hrcropped = channel.crop(hrarea)
            hrcropped.save(newdir + '\\HR_' + channel_dict[channelno] +'_' + numi +'.png')
            channelno=channelno+1

        ##create fake cloud image for use in DeepSUM
        hrarea = (0, 0, 512, 512)
        hrcloud = Image.open(cloudtemplate)
        hrcroppedcloud = hrcloud.crop(hrarea)
        hrcroppedcloud.save(newdir + '\\fakecloud.png')

        for imagename in sentinel:
            if imagename[-5:] != "cloud":
                channelno=0
                for channel in sentinel[imagename].split():
                    cropped = channel.crop(area)
                    lrtopleft = channel.getpixel((randheight + cropheight, randwidth))
                    lrtopright = channel.getpixel((randheight + cropheight, randwidth + cropwidth))
                    lrbottomleft = channel.getpixel((randheight, randwidth))
                    lrbottomright = channel.getpixel((randheight, randwidth + cropwidth))

                    if lrtopleft !=-1 and lrtopright !=-1 and lrbottomleft !=-1 and lrbottomright !=-1:
                        cropped.save(newdir + '\\' + channel_dict[channelno] + '_' + numi + '_'  + imagename  + '.png')
                    channelno=channelno+1
                cloudimg = Image.open(dir + "\\" + imagename[:-4] + "_cloud.png")

                ##converts cropped cloud to 1bit and reverses this
                croppedcloud = sentinel[imagename[:-4] + "_cloud"].crop(area)
                cloudcropped_np = np.array(croppedcloud)
                where0 = np.where(cloudcropped_np == 0)
                where1 = np.where(cloudcropped_np > 0)## reverse array so clear values are 0 and cloud values are 1
                cloudcropped_np[where1] = 255
                cloudcropped_np[where0] = 0
                new_im = Image.fromarray(cloudcropped_np)
                croppedcloud1bt = new_im.convert('1')

                croppedcloud1bt.save(newdir + '\\cloud_' + numi + '_' + imagename + '.png')

                cropped = sentinel[imagename].crop(area)
                cropped.save(newdir + '\\ALL_' + numi + '_'  + imagename  + '.png')
        i = i+1


