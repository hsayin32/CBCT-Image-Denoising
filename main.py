# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:08:18 2021

@author: user
"""
import numpy as np
import cv2 as cv
from PIL import Image
import os #
from scipy import signal
from pylab import*

x=str("koronal")

def resimler(yol):
    return [os.path.join(yol,f) for f in os.listdir(yol)]
filelist=resimler("D:/........./codes/source/patients/patient1/jpg/"+x+"/8bit/")

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)


def MMGnet(img,me,med,gua):
    new_image=np.zeros((len(img),len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            new_image[i][j]=(((me[i][j])**(2))+((med[i][j])**(2))+((gua[i][j])**(2)))**(0.5)
            
            
    return new_image*255/new_image.max()


m=0
for z,resim in enumerate(filelist):
    img = np.array(Image.open(resim))
    #type1 median
    #img2 = cv.medianBlur(img,7)
    
    """
    #type2 MMG
    mean = cv.blur(img,(7,7))
    median = cv.medianBlur(img,7)
    guasian = cv.GaussianBlur(img,(7,7),0)
    mmg1=MMGnet(img,mean,median,guasian)
    img2=mmg1.astype(np.uint8)
    """
    #type3 bilateral
    img2 = cv.bilateralFilter(img,8,50,50)
    
    
    
    
    hist,bins = np.histogram(img2.flatten(),256,[30,60])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img3 = cdf[img]
    
    th, img4 = cv.threshold(img3, round(sum(cdf)/256)+5, 255, cv.THRESH_BINARY);
    img5 = cv.erode(img4,(5,5),iterations = 2)
                
    img7 = img3 | img5 
    #ROI process
    #axialroi
    roi = img7[55:215, 110:405]
    #koronalroi
    #roi = img7[200:415, 100:420]
    #sagitalroi
    #roi = img7[150:350, 90:350]
    
    cl1=adjust_gamma(roi, gamma=0.7)
    roi1=cl1
    
    #axial (160x295) koronal(215x320)  sagital(200x260)
    
    
    for i in range(0,215):
        for j in range(0,320):
            if roi1[i,j]>49  and roi1[i,j]<100:
                roi1[i,j]-=50
            if roi1[i,j]>99  and roi1[i,j]<150:
                roi1[i,j]-=100
            if roi1[i,j]>149  and roi1[i,j]<200:
                roi1[i,j]-=100
            if roi1[i,j]>199  and roi1[i,j]<219:
                roi1[i,j]-=150
            if roi1[i,j]>220:
                roi1[i,j]=255
                
    roi1 = cv.medianBlur(roi1,5)
    roi1 = cv.dilate(roi1,(3,3),iterations = 1)  

    cl2=adjust_gamma(roi1, gamma=1.4)
        
    img7[200:415, 100:420]=cl2
        
    cv.imwrite("D:/.............../codes/source/patients/patient1/bilateral_jpg/"+x+"/"+str(z+1)+".jpg",img7)
    print(str(z+1)+"   " +str(int(z*100/len(filelist)))+"%")
    
    
    
    
