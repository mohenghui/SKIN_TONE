import enum
import os
from cv2 import imread
from imageio import save
from matplotlib.pyplot import axis, clf
import numpy as np
import cv2
from utils import SkinMask1,SkinMask2, SkinMask3,RandomForest,SkinWhiten1,SkinWhiten2,SkinWhiten3,FindFiles
# label=np.loadtxt(dtype=np.int32)
input_data=np.genfromtxt('./data/Skin_NonSkin.txt',dtype=np.int32)
save_path="./skin_tone_val/"
image_path="./skin_tone_val/image/"
mask1_path=os.path.join(save_path,"mask1")
mask2_path=os.path.join(save_path,"mask2")
mask3_path=os.path.join(save_path,"mask3")
skin_model=RandomForest(input_data)
image_name=os.listdir(image_path)
#获得mask图片
for idx,name in enumerate(image_name):
    og_img=cv2.imread(os.path.join(image_path,name))
    SkinMask1(og_img,img_name=name,save_path=mask1_path)
    SkinMask2(og_img,img_name=name,save_path=mask2_path)
    SkinMask3(roi=og_img,clf=skin_model,img_name=name,save_path=mask3_path)

mask_file=[]
mask_file=FindFiles(save_path,mask_file)
for idx,name in enumerate(mask_file):
    og_img=cv2.imread(os.path.join(image_path,os.path.basename(name)))
    mask_img=cv2.imread(name)
    name_split=name.split('/')
    save_path=os.path.join(name_split[-3],'ouput'+name_split[-2][-1],name_split[-1])
    #调整hsv的亮度通道
    # SkinWhiten1(og_img,mask_img,save_path)
    #logarithmic Curve
    # SkinWhiten2(og_img,mask_img,save_path)
    #打表增亮
    SkinWhiten3(og_img,mask_img,save_path)