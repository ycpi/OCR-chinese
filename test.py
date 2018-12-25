#!/usr/bin/env python
# -- coding: utf-8 --
from yolo3_predict import YOLO
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
from yolo3.utils import sort_box
from crnn.crnn import crnn
from multiprocessing import Process, Queue, Pool

# 控制detection网络占用GPU的大小
# import tensorflow as tf   
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

def detect_img_test():
    yolo = YOLO()
    img_path = 'invoice.jpg'
    image = Image.open(img_path)
    r_image, r_box = yolo.detect_image(image)
    yolo.close_session()
    r_image.save('output.jpg')
    # im_crop = image.crop(r_box[0])
    # im_crop.save('crop.jpg')


def detect_img(image_path,queue):
    yolo = YOLO()
    img_path = image_path
    image = Image.open(img_path)
    r_image, r_box = yolo.detect_image(image)
    yolo.close_session()
    queue.put(r_box)

def recognition_img(image_path):
    #通过多进程来混合使用pytorch和keras
    q = Queue()

    p = Process(target=detect_img, args=(image_path,q))
    p.start()
    p.join()

    # detect_img(image_path,q)

    r_box = q.get()
    
    image = Image.open(image_path)
    r_box = sort_box(r_box)
    results = []
    crnn_test = crnn()

    tmp = np.array(image)

    thickness = (image.size[0] + image.size[1]) // 512 

    for box in r_box: #画框
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])
        x3, y3 = (box[4], box[5])
        x4, y4 = (box[6], box[7]) 

        c=(0,0,0) #color

        cv2.line(tmp,(int(x1),int(y1)),(int(x2),int(y2)),c,thickness)
        cv2.line(tmp,(int(x2),int(y2)),(int(x3),int(y3)),c,thickness)
        cv2.line(tmp,(int(x3),int(y3)),(int(x4),int(y4)),c,thickness)
        cv2.line(tmp,(int(x4),int(y4)),(int(x1),int(y1)),c,thickness)


    for i, box in enumerate(r_box):
        degree,w,h,cx,cy = solve_img(box)
        partImg,newW,newH = rotate_cut_img(image,degree,box,w,h)  #按照box的倾斜角度剪切图像
        # plt.imshow(partImg)
        # plt.show()
        im_crop_gray = partImg.convert('L')
        crnn_pre = crnn_test.crnnOcr(im_crop_gray)
        if crnn_pre.strip() != u'':
            print(crnn_pre)
            results.append({
                'left': box[0],
                'top': box[1],
                'right': box[4],
                'bottom': box[5],
                'label': crnn_pre,
            })
        # else:
        #     r_box = np.delete(r_box, i, axis=0)

    # r_box = np.column_stack((r_box, results))

    image = Image.fromarray(tmp)

    font = ImageFont.truetype(font='font/华文仿宋.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
    
    for box in results:
        draw = ImageDraw.Draw(image)  # 画图

        left=box['left']
        top=box['top']
        right=box['right']
        bottom=box['bottom']
        label = box['label']
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        label_size = draw.textsize(label, font)

        if top - label_size[1] >= 0:  # 标签文字
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            # draw.rectangle(# 画框
            #     [left + i, top + i, right - i, bottom - i], outline=(0, 0, 0))
            # draw.rectangle(# 文字背景
            #     [tuple(text_origin), tuple(text_origin + label_size)], outline=(0, 0, 0))
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
        del draw

    image.save('output444.jpg')
    
def solve_img(box):
    x1,y1,x2,y2,x3,y3,x4,y4= box[:8]
    cx = (x1+x3+x2+x4)/4.0
    cy = (y1+y3+y4+y2)/4.0  
    w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
    h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2   

    sinA = (h*(x1-cx)-w*(y1 -cy))*1.0/(h*h+w*w)*2
    angle = np.arcsin(sinA)
    return angle,w,h,cx,cy


def rotate_cut_img(im,degree,box,w,h):
    x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
    x_center,y_center = np.mean([x1,x2,x3,x4]),np.mean([y1,y2,y3,y4])
    degree_ = degree*180.0/np.pi
    
    box = (max(1,x_center-w/2)##xmin
           ,y_center-h/2,##ymin
           min(x_center+w/2,im.size[0]-1)##xmax
           ,y_center+h/2)##ymax
 
    newW = box[2]-box[0]
    newH = box[3]-box[1]
    tmpImg = im.rotate(degree_,center=(x_center,y_center)).crop(box)
    return tmpImg,newW,newH

if __name__ == '__main__':
    # detect_img_test()
    
    recognition_img('invoice4.jpg')
