from unittest import result
import cv2
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import os
import math
def threshold_image(image):
    ret, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold

def makedirR(c_path):
    if '.'not in c_path and not os.path.exists(c_path):
        os.mkdir(c_path)
    elif '.' in c_path :
        tmp='/'.join(c_path.split('/')[:-1])
        if  not os.path.exists(tmp):
            os.mkdir(tmp)

#遍历法
def SkinMask1(roi,img_name,save_path):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # 转换到RGB空间
    (R, G, B) = cv2.split(rgb)  # 获取图像每个像素点的RGB的值，即将一个二维矩阵拆成三个二维矩阵
    skin_mask = np.zeros(R.shape, dtype=np.uint8)  # 掩膜
    (x, y) = R.shape  # 获取图像的像素点的坐标范围
    for i in range(0, x):
        for j in range(0, y):
            # 判断条件，不在肤色范围内则将掩膜设为黑色，即255
            if (abs(R[i][j] - G[i][j]) > 15) and (R[i][j] > G[i][j]) and (R[i][j] > B[i][j]):
                if (R[i][j] > 95) and (G[i][j] > 40) and (B[i][j] > 20) \
                        and (max(R[i][j], G[i][j], B[i][j]) - min(R[i][j], G[i][j], B[i][j]) > 15):
                    skin_mask[i][j] = 255
                elif (R[i][j] > 220) and (G[i][j] > 210) and (B[i][j] > 170):
                    skin_mask[i][j] = 255
    makedirR(save_path)
    save_path=os.path.join(save_path,img_name)
    cv2.imwrite(save_path,skin_mask)

# YCrCb之Cr分量 + OTSU二值化
def SkinMask2(roi,img_name,save_path):
    yCrCb=cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    (y,cr,cb)=cv2.split(yCrCb)
    img_cr=cv2.GaussianBlur(cr,(3,3),0) #对cr通道进行高斯滤波
    skin_mask=threshold_image(img_cr)
    makedirR(save_path)
    save_path=os.path.join(save_path,img_name)
    cv2.imwrite(save_path,skin_mask)

#随机森林
def SkinMask3(roi,clf,img_name,save_path):
    data = np.reshape(roi, (roi.shape[0] * roi.shape[1], 3))
    data = np.reshape(data, (data.shape[0], 1, 3))
    data= cv2.cvtColor(np.uint8(data), cv2.COLOR_BGR2HSV)
    roi_hsv= np.reshape(data,(data.shape[0],3))
    predictedLabels= clf.predict(roi_hsv)
    imgLabels= np.reshape(predictedLabels,(roi.shape[0],roi.shape[1],1))
    makedirR(save_path)
    save_path=os.path.join(save_path,img_name)
    cv2.imwrite(save_path,((-(imgLabels-1)+1)*255))
def RandomForest(input_data):
    data = input_data
    labels= data[:,3]
    data= data[:,0:3]
    data = np.reshape(data, (data.shape[0], 1, 3))
    data = cv2.cvtColor(np.uint8(data), cv2.COLOR_BGR2HSV)
    data = np.reshape(data, (data.shape[0], 3))
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.2, random_state=42)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(trainData, trainLabels)
    return clf
def FindFiles(path,result):
# 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 循环判断每个元素是否是文件夹还是文件，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path) and file!="image":
            FindFiles(cur_path,result)
        # if len(file):
        elif os.path.isfile(cur_path):
            result.append(os.path.join(path,file))
    return result
def NormalImage(img,mask):
    img=img.astype('float32')/255.0
    mask=mask.astype('float32')/255.0
    return img,mask
def SkinWhiten1(img,mask,path,rate=0.15):
    # npr_img,nor_mask=NormalImage(img,mask)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,-1] = np.minimum(
    img_hsv[:,:,-1] + img_hsv[:,:,-1] * mask[:,:,-1] * rate,
        255).astype('uint8')
    img_beauty = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    makedirR(path)
    cv2.imwrite(path,img_beauty)
def SkinWhiten2(img,mask,path,beta=2.0):
    img,mask=NormalImage(img,mask)
    tmp=img*(1-mask)+np.log(np.dot(img * mask , (beta - 1)) + 1) / np.log(beta)
    img=np.clip(tmp,0,1)
    img=(img*255.0).astype("uint8")
    makedirR(path)
    cv2.imwrite(path,img)
def SkinWhiten3(img,mask,path,rate=0.0.7):
    Color_list = [
	1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39,
	41, 43, 44, 46, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 73, 74,
	76, 78, 79, 81, 83, 84, 86, 87, 89, 91, 92, 94, 95, 97, 99, 100, 102, 103, 105,
	106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 128,
	130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150,
	151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
	171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
	188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
	204, 205, 205, 206, 207, 208, 209, 210, 211, 211, 212, 213, 214, 215, 215, 216,
	217, 218, 219, 219, 220, 221, 222, 222, 223, 224, 224, 225, 226, 226, 227, 228,
	228, 229, 230, 230, 231, 232, 232, 233, 233, 234, 235, 235, 236, 236, 237, 237,
	238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 244, 245,
	245, 246, 246, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 250,
	251, 251, 251, 251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254,
	254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 256]
    # img,mask=NormalImage(img,mask)
    img = cv2.bilateralFilter(img, 19, 75, 75)
    h, w, _ = img.shape
    img_copy = img.copy()*mask
    for i in range(h):
        for j in range(w):
            b = img_copy[i, j, 0]
            g = img_copy[i, j, 1]
            r = img_copy[i, j, 2]
            img_copy[i, j, 0] = Color_list[b]
            img_copy[i, j, 1] = Color_list[g]
            img_copy[i, j, 2] = Color_list[r]
    img = np.minimum(
    img + img_copy * rate,
        255).astype('uint8')
    makedirR(path)
    cv2.imwrite(path,img)
    