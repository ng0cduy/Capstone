######################################################################
# set the optimizer
from __future__ import print_function, division
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import albumentations as A
def param_updates (model_feature,feature_extract_state):
    params_to_update = model_feature.parameters()
    print("Params to learn:")
    if feature_extract_state:
        params_to_update = []
        for name,param in model_feature.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_feature.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return params_to_update  
# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting == True:
        for param in model.parameters():
            param.requires_grad = False

def init_model (model_name,num_classes,feature_extract=True):
    model_ft = None
    input_size =224
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name =="alexnet":
        model_ft = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224
    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft,feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    return model_ft

def perf_measure(y_actual, y_pred,pos_label=0):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==pos_label:
           TP += 1
        if y_pred[i]==pos_label and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]!=pos_label:
           TN += 1
        if y_pred[i]!=pos_label and y_actual[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)
def fscore_compute(tp=0,fp=0,tn=0,fn=0):
    if tp + fp ==0:
        return (0,0,0)
    else:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fscore = 2*precision*recall/(precision+recall)
        return (precision,recall,fscore)
def rate_compute(tp,fp,tn,fn):
    '''
    compute FNR,FPR,TPR,TNR
    return FNR: Miss Detection Rate, FPR: False Alarm Rate
    '''
    TPR = tp/(tp+fn)
    FNR = fn/(tp+fn)
    FPR = fp/(fp+tn)
    TNR = tn/(fp+tn)
    return (TPR,FNR,FPR,TNR)
