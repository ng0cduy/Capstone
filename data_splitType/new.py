if __name__ == '__main__':
    from conv_func import *
    import argparse
    plt.ion()   # interactive mode

    ######################################################################
    # Load Data
    # ---------
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_transforms_A = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.1,hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.1,hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    '''
    BURR : bur
    DUST : dust
    MSTN : stain
    MBLS : belly
    MLKS : locks
    PCS  : skirting
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirr",help='add data path',type=str)
    parser.add_argument("model_name",help='add model name',type=str)
    args = parser.parse_args()
    data_dir = args.data_dirr
    # data_dir = 'fleece_data_BURR'  #del later
    print(data_dir)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    image_datasets_A = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms_A[x])
                    for x in ['train', 'val']}  
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
    #                                              shuffle=True, num_workers=4)
    #               for x in ['train', 'val']}
    data_concat_train = torch.utils.data.ConcatDataset([image_datasets['train'],image_datasets_A['train']])
    data_concat_val = torch.utils.data.ConcatDataset([image_datasets['val'],image_datasets_A['val']])
    image_datasets_new = image_datasets_new = {'train': data_concat_train, 'val':data_concat_val};
    dataloaders_new = {x: torch.utils.data.DataLoader(image_datasets_new[x], batch_size=8,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets_new[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0")



    inputs, classes = next(iter(dataloaders_new['train']))
    out = torchvision.utils.make_grid(inputs)

    data_total_size = sum(dataset_sizes.values())
    
    ######################################################################
    # Training the model
    # ------------------
    #

    def train_model(model1,model2, criterion1,criterion2, optimizer1,optimizer2, scheduler1,scheduler2, num_epochs=25):
        since = time.time()
        best_model_wts1 = copy.deepcopy(model1.state_dict())
        best_model_wts2 = copy.deepcopy(model2.state_dict())
        best_acc1 = 0.0
        best_acc2 = 0.0
        val_acc_history1 = []
        f_score_hist1 = []
        val_acc_history2 = []
        f_score_hist2 = []
        for epoch in range(num_epochs):
            print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 100)
            pred_1 =np.array([])
            pred_1 = torch.LongTensor(pred_1)
            pred_1 = pred_1.to(device)
            true_1 = np.array([])
            true_1 = torch.LongTensor(true_1)
            true_1 = true_1.to(device)
            
            pred_2 =np.array([])
            pred_2 = torch.LongTensor(pred_2)
            pred_2 = pred_2.to(device)
            true_2 = np.array([])
            true_2 = torch.LongTensor(true_2)
            true_2 = true_2.to(device)
            TP1 = 0     
            FP1 = 0
            TN1 = 0
            FN1 = 0
            TP2 = 0     
            FP2 = 0
            TN2 = 0
            FN2 = 0
            total_running_correct1 = 0;
            total_running_correct2 = 0;
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model1.train()  # Set model to training mode
                    model2.train()
                else:
                    model1.eval()   # Set model to evaluate mode
                    model2.eval()

                running_loss1 = 0.0
                running_corrects1 = 0
                running_loss2 = 0.0
                running_corrects2 = 0
                running_not_contam = 0
                # Iterate over data.
                for inputs, labels in dataloaders_new[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs1 = model1(inputs)
                        outputs2 = model2(inputs)
                        _, preds1 = torch.max(outputs1, 1)
                        _, preds2 = torch.max(outputs2, 1)
                        loss1 = criterion1(outputs1, labels)
                        loss2 = criterion2(outputs2, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss1.backward()
                            loss2.backward()
                            optimizer1.step()
                            optimizer2.step()

                    # statistics
                    running_loss1 += loss1.item() * inputs.size(0)
                    running_corrects1 += torch.sum(preds1 == labels.data)
                    running_loss2 += loss2.item() * inputs.size(0)
                    running_corrects2 += torch.sum(preds2 == labels.data)
                    # print ('P',preds)
                    # print('T',labels.data)
                    pred_1 = torch.cat([pred_1,preds1])
                    true_1 = torch.cat([true_1,labels.data])
                    
                    pred_2 = torch.cat([pred_2,preds2])
                    true_2 = torch.cat([true_2,labels.data])
                    total_running_correct1 +=running_corrects1.double()
                    total_running_correct2 +=running_corrects2.double()
                if phase == 'train':
                    scheduler1.step()
                    scheduler2.step()

                epoch_loss = running_loss1 / dataset_sizes[phase]
                epoch_acc1 = running_corrects1.double() / dataset_sizes[phase]
                epoch_loss2 = running_loss2 / dataset_sizes[phase]
                epoch_acc2 = running_corrects2.double() / dataset_sizes[phase]
                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                #     phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val':
                    if epoch_acc1 > best_acc1:
                        best_acc1 = epoch_acc1
                    if epoch_acc2 > best_acc2:
                        best_acc2 = epoch_acc2

                if phase =='val' and epoch == num_epochs -1:
                    best_model_wts1 = copy.deepcopy(model1.state_dict())
                    best_model_wts2 = copy.deepcopy(model2.state_dict())
                if phase == 'val':
                    val_acc_history1.append(epoch_acc1)
                    val_acc_history2.append(epoch_acc2)
            pred_1 = pred_1.cpu().numpy()
            true_1 = true_1.cpu().numpy()
            pred_2 = pred_2.cpu().numpy()
            true_2 = true_2.cpu().numpy()
            TP1, FP1, TN1, FN1 = perf_measure(true_1,pred_1,pos_label=0)
            TP2, FP2, TN2, FN2 = perf_measure(true_2,pred_2,pos_label=0)
            print(f"tp: {TP1} fp: {FP1} tn: {TN1} fn: {FN1}")
            print(f"tp: {TP2} fp: {FP2} tn: {TN2} fn: {FN2}")
            precision1,recall1,f_beta_score1,support1 = precision_recall_fscore_support(true_1, pred_1,pos_label = 0,average='binary')
            precision2,recall2,f_beta_score2,support2 = precision_recall_fscore_support(true_2, pred_2,pos_label = 0,average='binary')
            f_score_hist1.append(f_beta_score1)
            f_score_hist2.append(f_beta_score2)
            print('Precision {:.4f} Recall {:.4f} F_score {:.4f}'.format(precision1,recall1,f_beta_score1))
            print('Precision {:.4f} Recall {:.4f} F_score {:.4f}'.format(precision2,recall2,f_beta_score2))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model1.load_state_dict(best_model_wts1)
        model2.load_state_dict(best_model_wts2)
        return model1,model2,val_acc_history1,val_acc_history2,f_score_hist1,f_score_hist2,time_elapsed

    ######################################################################
    # Visualizing the model predictions
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def visualize_model(model, num_images=4,model_name=''):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(18,9))
        out_preds = ''
        out_label = ''
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders_new['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('inputs',inputs)
                outputs = model(inputs)
                # print('output',outputs)
                _, preds = torch.max(outputs, 1)
                for j in range(inputs.size()[0]):
                    if class_names[preds[j]] == 'contaminated':
                        out_preds = 'Contam'
                    else:
                        out_preds = 'Clean'
                    if class_names[labels.data[j]] == 'contaminated':
                        out_label = 'Contam'
                    else:
                        out_label = 'Clean'
                    images_so_far += 1
                    col = 4
                    ax = plt.subplot(num_images//col, col, images_so_far)
                    ax.axis('off')
                    ax.set_title('P: {} T: {}'.format(out_preds,out_label))
                    # ax.set_title('P: {} T: {}'.format(class_names[preds[j]],class_names[labels.data[j]]))
                    imshow(inputs.cpu().data[j])
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    ####################################TEST ZONE##########################
    def visualize_model1(model1,model2, num_images=16,model_name=''):
        was_training1 = model1.training
        was_training2 = model2.training
        model1.eval()
        model2.eval()
        images_so_far = 0
        images_so_far1 = 0
        f1 = plt.figure(1,figsize=(18,9))
        f2  = plt.figure(2,figsize=(18,9))
        out_preds1 = ''
        out_preds2 = ''
        out_label = ''
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders_new['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('inputs',inputs)
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                # print('output',outputs)
                _, preds1 = torch.max(outputs1, 1)
                _, preds2 = torch.max(outputs2, 1)
                
                for j in range(inputs.size()[0]):
                    if class_names[preds1[j]] == 'contaminated':
                        out_preds1 = 'Contam'
                    else:
                        out_preds1 = 'Clean'
                    if class_names[preds2[j]] == 'contaminated':
                        out_preds2 = 'Contam'
                    else:
                        out_preds2 = 'Clean'

                    if class_names[labels.data[j]] == 'contaminated':
                        out_label = 'Contam'
                    else:
                        out_label = 'Clean'
                    
                    images_so_far += 1
                    images_so_far1 += 1
                    col = 4
                    plt.figure(1)
                    ax = plt.subplot(num_images//col, col, images_so_far)
                    ax.axis('off')
                    ax.set_title('P: {} T: {}'.format(out_preds1,out_label))
                    imshow(inputs.cpu().data[j])
                    plt.figure(2)
                    ax = plt.subplot(num_images//col, col, images_so_far)
                    ax.axis('off')
                    ax.set_title('P: {} T: {}'.format(out_preds2,out_label))
                    imshow(inputs.cpu().data[j])
                    if images_so_far == num_images:
                        model1.train(mode=was_training1)
                        model2.train(mode=was_training2)
                        return
            model1.train(mode=was_training1) 
            model2.train(mode=was_training2)           
    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    #
    # model_names = ['alexnet','resnet','squeezenet','densenet']
    num_e = 25


    args = parser.parse_args()
    model_name_ = args.model_name
    model_ft_extract = False
    # model_name_ = 'alexnet' #del-later
    model_ft = init_model(model_name=model_name_,num_classes=2,feature_extract=model_ft_extract)
    print(model_name_)
    model_ft = model_ft.to(device)

    criterion_ft = nn.CrossEntropyLoss()
    param_ft = param_updates(model_ft,model_ft_extract)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(param_ft,lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate


    # model_ft,ohist,fscore_hist,time_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                     num_epochs=num_e)
    #####################################################################
    

    # visualize_model(model_ft,model_name=model_name_)


    # ######################################################################
    # # ConvNet as fixed feature extractor
    # # ----------------------------------

    conv_ft_extract = True
    model_conv = init_model(model_name=model_name_,num_classes=2,feature_extract=conv_ft_extract)

    model_conv = model_conv.to(device)

    criterion_conv = nn.CrossEntropyLoss()
    param_conv = param_updates(model_conv,conv_ft_extract)
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(param_conv, lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_conv = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    # model_conv,fhist,fscore_hist_f,time_ffe = train_model(model_conv, criterion, optimizer_conv,
    #                         exp_lr_scheduler, num_epochs=num_e)

    # TRAIN AND EVALUATE BOTH
    model_ft,model_conv,val_acc_history1,val_acc_history2,f_score_hist1,f_score_hist2,time_elapsed = train_model(model_ft,model_conv,criterion_ft,criterion_conv,optimizer_ft,
    optimizer_conv,exp_lr_scheduler_ft,exp_lr_scheduler_conv,num_e)

    # ######################################################################
    # #
    # Visualize some images
    # visualize_model(model_ft,model_name=model_name_)
    # visualize_model(model_conv,model_name=model_name_)
    visualize_model1(model_ft,model_conv)
    # ploting graph
    # epochs =num_e
    # ohist_ = []
    # fscore_hist_ = []
    # fscore_hist_f_ = []
    # fscore_hist_ = [h for h in fscore_hist]
    # fscore_hist_f_ = [h1 for h1 in fscore_hist_f]
    # print(time_ft)
    # print(time_ffe)
    # fig1 = plt.figure(num=model_name_,figsize=(18,9))
    # plt.title(f"F_score vs. Number of Training Epochs {model_name_}")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("F_score")
    # plt.plot(range(1,epochs+1),fscore_hist_,label="Fine_tunning")
    # plt.plot(range(1,epochs+1),fscore_hist_f_,label="Fixed Feature Extractor")
    # plt.ylim((0,1.))
    # plt.xticks(np.arange(1,epochs+1,1.0))
    # plt.legend()


    # fhist_ = []
    # ohist_ = [h.cpu().numpy() for h in ohist]
    # fhist_ = [h.cpu().numpy() for h in fhist]
    # fig1 = plt.figure(num=model_name_,figsize=(18,9))
    # plt.title(f"Validation Accuracy vs. Number of Training Epochs {model_name_}")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(range(1,epochs+1),ohist_,label="Finetuning the convnet")
    # plt.plot(range(1,epochs+1),fhist_,label="ConvNet as fixed feature extractor")
    # plt.ylim((0,1.))
    # plt.xticks(np.arange(1, epochs+1, 1.0))
    # plt.legend()
    plt.ioff()
    plt.show()
