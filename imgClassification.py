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
plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'fleece_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
# print(dataloaders['train'])
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# set the optimizer
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


inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)

data_total_size = sum(dataset_sizes.values())
  
######################################################################
# Training the model
# ------------------
#

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
        pred_ =np.array([])
        pred_ = torch.LongTensor(pred_)
        pred_ = pred_.to(device)
        true_ = np.array([])
        true_ = torch.LongTensor(true_)
        true_ = true_.to(device)
        TP = 0;
        FP = 0;
        TN = 0;
        FN = 0;
        total_running_correct = 0;
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_not_contam = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print ('P',preds)
                # print('T',labels.data)
                pred_ = torch.cat([pred_,preds])
                true_ = torch.cat([true_,labels.data])
                total_running_correct +=running_corrects.double()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        pred_ = pred_.cpu().numpy()
        true_ = true_.cpu().numpy()
        TP, FP, TN, FN = perf_measure(true_,pred_,pos_label=0)
        print(f"tp: {TP} fp: {FP} tn: {TN} fn: {FN}")
        precision,recall,f_beta_score,support = precision_recall_fscore_support(true_, pred_,pos_label = 0,average='binary')
        # print(classification_report(true_, pred_,target_names=['contaminant','not_contaminant'],zero_division=0,digits=4))
        print('Precision {:.4f} Recall {:.4f} F_score {:.4f}'.format(precision,recall,f_beta_score))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,val_acc_history

######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def visualize_model(model, num_images=12,model_name=''):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(18,9))
    out_preds = ''
    out_label = ''
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if class_names[preds[j]] == 'contaminated':
                    out_preds = 'Contam'
                else:
                    out_preds = 'Not Contam'
                if class_names[labels.data[j]] == 'contaminated':
                    out_label = 'Contam'
                else:
                    out_label = 'Not Contam'
                images_so_far += 1
                ax = plt.subplot(num_images//3, 3, images_so_far)
                ax.axis('off')
                ax.set_title('P: {} T: {}'.format(out_preds,out_label))
                # ax.set_title('P: {} T: {}'.format(class_names[preds[j]],class_names[labels.data[j]]))
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
######################################################################
# Finetuning the convnet
# ----------------------
#
#
model_names = ['alexnet','resnet','squeezenet','densenet']
num_e = 25
# model_name_ = "densenet"
for model_name_ in model_names:
    model_ft_extract = False
    model_ft = init_model(model_name=model_name_,num_classes=2,feature_extract=model_ft_extract)
    print(model_name_)


    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    param_ft = param_updates(model_ft,model_ft_extract)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(param_ft,lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
 

    model_ft,ohist = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=num_e)
    ######################################################################
    #

    visualize_model(model_ft,model_name=model_name_)


    ######################################################################
    # ConvNet as fixed feature extractor
    # ----------------------------------

    conv_ft_extract = True
    model_conv = init_model(model_name=model_name_,num_classes=2,feature_extract=conv_ft_extract)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    param_conv = param_updates(model_conv,conv_ft_extract)
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(param_conv, lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    model_conv,fhist = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=num_e)

    ######################################################################
    #
    epochs =num_e
    visualize_model(model_conv,model_name=model_name_)
    ohist_ = []
    fhist_ = []
    ohist_ = [h.cpu().numpy() for h in ohist]
    fhist_ = [h.cpu().numpy() for h in fhist]
    fig1 = plt.figure(num=model_name_,figsize=(18,9))
    plt.title(f"Validation Accuracy vs. Number of Training Epochs {model_name_}")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,epochs+1),ohist_,label="Finetuning the convnet")
    plt.plot(range(1,epochs+1),fhist_,label="ConvNet as fixed feature extractor")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, epochs+1, 1.0))
    plt.legend()
    plt.ioff()
    plt.show()
