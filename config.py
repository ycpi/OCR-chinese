import os
pwd = os.getcwd()

yoloAnchors = os.path.join(pwd,"configs","yolo_anchors.txt")
yoloWeights = os.path.join(pwd,"model_data","text.h5")
yoloClasses = os.path.join(pwd,"configs","text.txt")

IMGSIZE = (416,416)## yolo3 输入图像尺寸,必须为32的倍数,(416,416),(960,960)
##是否启用LSTM crnn模型
DETECTANGLE=False##是否进行文字方向检测
LSTMFLAG = False##OCR模型是否调用LSTM层
GPU = True##OCR 是否启用GPU
GPUID=0##调用GPU序号
chinsesModel = True##模型选择 True:中英文模型 False:英文模型

if chinsesModel:
    if LSTMFLAG:
        ocrModel  = os.path.join(pwd,"model_data","ocr-lstm.pth")
    else:
        ocrModel = os.path.join(pwd,"model_data","ocr-dense.pth")
else:
        LSTMFLAG=True
        ocrModel = os.path.join(pwd,"model_data","ocr-english.pth")
