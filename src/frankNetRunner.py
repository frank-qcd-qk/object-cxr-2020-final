from ctypes import *
import random
import torch
import numpy as np
import cv2
import darknetCore.darknet as darknet
from darknetCore.dataPacker import packData
import os
import csv
import pandas as pd
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.backends import cudnn
import pickle
import os

parser = argparse.ArgumentParser(description='Run Frank submission')
parser.add_argument('input_csv', default='image_path.csv', metavar='INPUT_CSV',
                    type=str, help="input-data-csv-filename")
parser.add_argument('classification_csv', default='classification.csv', metavar='CLASSIFICATION_CSV',
                    type=str, help="output-classification-prediction-csv-path")
parser.add_argument('localization_csv', default='localization.csv', metavar='LOCALIZATION_CSV',
                    type=str, help='output-localization-prediction-csv-path')

args = parser.parse_args()

netMain = None
metaMain = None
altNames = None

classificationWriter = None
localizationWriter = None
OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

detectionThreashold = 0.1
predictionOverRide = False
predictionMultiboxOverride = False
image2DetectList = None

img_size = 600
num_classes = 2

tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
model1 = model2 = None

configPath = "/workENV/src/frankNet/frankNet.cfg"
weightPath = "/workENV/src/frankNet/frankNet.weights"
metaPath = "/workENV/src/frankNet/obj.data"

packedData = None


def rcnnInit1():
    global model1
    model1 = fasterrcnn_resnet50_fpn(
        num_classes=num_classes, pretrained_backbone=False, pretrained=False)
    try:
        model1.load_state_dict(torch.load('/workENV/src/frankNet/RCNN1.pt'))
    except Exception:
        model1.load_state_dict(torch.load('src/frankNet/RCNN1.pt'))
    model1.eval()
    print("Load RCNN1 Complete!")


def rcnnInit2():
    global model2, packedData
    try:
        with open('src/frankNet/RCNN2.pt', 'rb') as modelFile:
            model2 = pickle.load(modelFile)
        with open('src/frankNet/frankNetR2.cfg', 'rb') as configFile:
            cfg = pickle.load(configFile)
    except Exception:
        with open('/workENV/src/frankNet/RCNN2.pt', 'rb') as modelFile:
            model2 = pickle.load(modelFile)
        with open('/workENV/src/frankNet/frankNetR2.cfg', 'rb') as configFile:
            cfg = pickle.load(configFile)
    packedData = packData(args.input_csv, cfg)
    model2.eval()
    print("Load RCNN2 Complete!")


def evalRCNN1(modelInput, imagePath):
    imageObject = tfms(Image.open(imagePath).convert('RGB'))
    detectionOut = modelInput([imageObject])
    return detectionOut


def yoloInit():
    global metaMain, netMain, altNames, configPath, weightPath, metaPath
    #! File path checker
    if not os.path.exists(configPath):
        configPath = "src/frankNet/frankNet.cfg"
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        weightPath = "src/frankNet/frankNet.weights"
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        metaPath = "src/frankNet/localobj.data"
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath)+"`")
    #! Net loader
    if netMain is None:
        try:
            netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        except Exception:
            raise ValueError(
                "Unable to initiate netMain. Check your config file!")
    if metaMain is None:
        try:
            metaMain = darknet.load_meta(metaPath.encode("ascii"))
        except Exception:
            raise ValueError(
                "Unable to initiate metaMain. Check your .data file!")
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    raise ValueError(
                        "Unable to initate Network. Check your weight file")
        except Exception:
            raise ValueError(
                "Unable to initate Network. Check your weight file")
    print()
    print("Load Yolo Network Complete.")

# Subcall for detecting a single image.


def singleImageDetection(imageIn):
    global detectionThreashold, model1, packedData
    yoloBBox = rcnn1BBox = rcnn2BBox = None
    counter = 0
    #! RCNN
    rcnn1BBox = evalRCNN1(model1, imageIn)
    #! Yolo
    imageIn = imageIn.encode("ascii")
    imageInput = darknet.load_image(imageIn, 0, 0)
    yoloBBox = darknet.detect_image(
        netMain, metaMain, imageInput, thresh=detectionThreashold)

    darknet.free_image(imageInput)
    return yoloBBox, rcnn1BBox


def fileIOInit():
    global localizationWriter, classificationWriter
    try:
        with open(args.classification_csv, 'w') as csvFile1:
            classificationWriter = csv.writer(csvFile1)
            classificationWriter.writerow(['image_path', 'prediction'])

        with open(args.localization_csv, 'w') as csvFile2:
            localizationWriter = csv.writer(csvFile2)
            localizationWriter.writerow(['image_path', 'prediction'])

    except Exception:
        raise IOError("Cannot initiate CSV file")

    print("File created successfully...")


def writer(which, what):
    if which == 1:
        with open(args.classification_csv, 'a') as csvFile1:
            classificationWriter = csv.writer(csvFile1)
            classificationWriter.writerow(what)
    else:
        with open(args.localization_csv, 'a') as csvFile2:
            localizationWriter = csv.writer(csvFile2)
            localizationWriter.writerow(what)


def getClassification(pathIn, yoloBBoxes, rcnn1BBox, rcnn2BBox):
    returner = [pathIn]
    localComparitor = []
    confidenceList = []
    #! Process Yolo
    if len(yoloBBoxes) != 0:
        for aDetection in yoloBBoxes:
            confidenceList.append(aDetection[1])
        localComparitor.append(max(confidenceList))
    else:
        localComparitor.append(0)

    #! Process rcnn1
    if len(rcnn1BBox[-1]['boxes']) == 0:
        localComparitor.append(0)
    else:
        localComparitor.append(torch.max(rcnn1BBox[-1]['scores']).tolist())

    #! Process rcnn2
    if len(rcnn2BBox) == 0:
        localComparitor.append(0)
    else:
        localComparitor.append(max(rcnn2BBox))
    localComparitor = [round(num, 4) for num in localComparitor]
    mostConfidence = round(max(localComparitor), 4)
    averageConfidence = round(sum(localComparitor)/len(localComparitor), 4)

    print("All Confidence: ", localComparitor)
    print("Most confidence: ", mostConfidence,
          " Average Confidence: ", averageConfidence)

    flag = True
    #! 0=yolo, 1=rcnn_main, 2= rcnn_aux
    # ? Previous high method:
    if (localComparitor[0] > 0.9 and localComparitor[1] > 0.95):
        print("All confidence large")
        towrite = str(1.0)
    else:
        towrite = str(round(localComparitor[1], 3))
    # # ? Current Voting Scheme:
    # towrite = None
    # if (averageConfidence > 0.92):
    #     print("All confidence large , force 1.0")
    #     towrite = averageConfidence
    # elif (averageConfidence < 0.08):
    #     print("All confidence small, force 0.0")
    #     towrite = averageConfidence
    #     flag = False
    # else:
    #     towrite = str(round(localComparitor[1], 3))
    returner.append(towrite)
    writer(1, returner)
    return flag


#! 2. classification bbox:
"""
image_path,confidence1 point1_x point1_y;confidence2 point2_x point2_y;confidence3 point3_x point3_y;confidence4 point4_x point4_y; ... confidenceK pointK_x pointK_y
valid_image/00004.jpg,0.25 500 700;0.51 320 1800;0.89 750 2200;0.49 1000 1200
"""


def getLocalization(pathIn, BBoxes, rcnn1BBox, rcnn2BBox, flag):
    returner = [pathIn]
    string2Write = ""
    # #! Write RCNN1 Detection
    # if len(rcnn1BBox) != 0 and flag:
    #     rcnn1Boxes = zip(rcnn1BBox[-1]['scores'].tolist(),
    #                      rcnn1BBox[-1]['boxes'].tolist())
    #     for aScore, aDetection in sorted(rcnn1Boxes):
    #         if aScore>0.1:
    #             center_x = int((aDetection[0] + aDetection[2]) / 2)
    #             center_y = int((aDetection[1] + aDetection[3]) / 2)
    #             string2Write += (str(round(aScore, 4)) +
    #                             " "+str(center_x+random.randint(-15, 15))+" "+str(center_y+random.randint(-15, 15))+";")
    #! Write RCNN2 Detection
    if flag:
        for aScore, aDetection in sorted(rcnn2BBox):
            if aScore>0.1:
                center_x = int((aDetection[0] + aDetection[2]) / 2)
                center_y = int((aDetection[1] + aDetection[3]) / 2)
                string2Write += (str(round(aScore, 4)) +
                                " "+str(center_x+random.randint(-5, 5))+" "+str(center_y+random.randint(-5, 5))+";")
        # remove dangling ";"
        string2Write = string2Write[:-1]
        returner.append(string2Write)
    else:
        returner.append('')
    writer(2, returner)


def ingestList():
    global image2DetectList
    image2DetectList = pd.read_csv(
        args.input_csv, header=None, na_filter=False, names=['path'])


def looper():
    global image2DetectList, model2
    filePaths, localizations = [], []
    with torch.no_grad():
        for anImage in packedData:
            anImagePath = anImage[0]['file_name']
            filePaths.append(anImage[0]['file_name'])
            boxOut = model2(anImage)[0]['instances']
            if len(boxOut) == 0:
                rcnn2Boxes = []
                rcnn2Scores = []
            else:
                rcnn2Boxes = boxOut.pred_boxes.tensor.cpu().numpy()
                rcnn2Scores = boxOut.scores.cpu().numpy()
            rcnn2BBox = zip(rcnn2Scores, rcnn2Boxes)
            print("ID: ", anImagePath)
            yoloBBox, rcnn1BBox = singleImageDetection(
                anImagePath)

            flags = getClassification(
                anImagePath, yoloBBox, rcnn1BBox, rcnn2Scores)
            getLocalization(anImagePath, yoloBBox, rcnn1BBox, rcnn2BBox, flags)
            print("============================================================")


def aucCalculator():
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    import matplotlib.pyplot as plt
    import subprocess
    print("++++++++++++++++++++++Metrics Report++++++++++++++++++++++")
    prediction_label = pd.read_csv(args.classification_csv, na_filter=False)
    pred = prediction_label.prediction
    pred = pred.replace(r'^\s*$', 0, regex=True)
    pred = pred.astype(float).values
    #print (pred)
    labels_dev = pd.read_csv('groundTruth.csv', na_filter=False)
    gt = labels_dev.annotation.astype(bool).astype(float).values
    # print(gt)
    # 1.
    acc = ((pred >= .5) == gt).mean()
    fpr, tpr, _ = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)

    #2, VisualIze
    fig, ax = plt.subplots(
        subplot_kw=dict(xlim=[0, 1], ylim=[0, 1], aspect='equal'),
        figsize=(6, 6)
    )
    ax.plot(fpr, tpr, label=f'ACC: {acc:.03}\nAUC: {roc_auc:.03}')
    _ = ax.legend(loc="lower right")
    _ = ax.set_title('ROC curve')
    ax.grid(linestyle='dashed')
    print("Report AUC: ", roc_auc, " Report ACC: ", acc)
    plt.show()


def frocCalculator():
    os.system('python3 src/froc.py loc.csv groundTruth.csv')


def runner():
    yoloInit()
    rcnnInit1()
    rcnnInit2()
    fileIOInit()
    ingestList()
    print()
    print("************************************************************************")
    looper()
    #print("Local run: calculating AUC:")
    try:
        aucCalculator()
        frocCalculator()
    except Exception:
        print("Cannot run evaluation.... It seems that I am in submission mode...")
        pass


if __name__ == "__main__":
    runner()
