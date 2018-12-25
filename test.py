#!/usr/bin/env python
# -- coding: utf-8 --
from yolo3_predict import YOLO
from PIL import Image, ImageFont, ImageDraw
from pylab import *
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
    img_path = 'invoice22.jpg'
    image = Image.open(img_path)
    r_image, r_box = yolo.detect_image(image)
    yolo.close_session()
    r_image.save('output22.jpg')
    # im_crop = image.crop(r_box[0])
    # im_crop.save('crop.jpg')


def detect_img(image_path,queue):
    yolo = YOLO()
    img_path = image_path
    image = Image.open(img_path)
    r_image, r_box = yolo.detect_image(image)
    yolo.close_session()
    r_image.save('output4.jpg')
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

    for i, box in enumerate(r_box):
        im_crop = image.crop(box)
        im_crop_gray = im_crop.convert('L')
        # imshow(im_crop)
        crnn_pre = crnn_test.crnnOcr(im_crop_gray)
        if crnn_pre.strip() != u'':
            print(crnn_pre)
            results.append({
                'left': box[0],
                'top': box[1],
                'right': box[2],
                'bottom': box[3],
                'label': crnn_pre,
            })
        # else:
        #     r_box = np.delete(r_box, i, axis=0)

    # r_box = np.column_stack((r_box, results))

    font = ImageFont.truetype(font='font/华文仿宋.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
    thickness = (image.size[0] + image.size[1]) // 512 
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
            draw.rectangle(# 画框
                [left + i, top + i, right - i, bottom - i], outline=(0, 0, 0))
            # draw.rectangle(# 文字背景
            #     [tuple(text_origin), tuple(text_origin + label_size)], outline=(0, 0, 0))
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
        del draw

    image.save('output111.jpg')
    



if __name__ == '__main__':
    # detect_img_test()
    
    recognition_img('invoice.jpg')
