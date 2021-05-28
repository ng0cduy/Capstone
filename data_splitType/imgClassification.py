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
    # data_dir = 'fleece_data_MSTN'
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

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        val_acc_history = []
        f_score_hist = []
        for epoch in range(num_epochs):
            print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
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
                for inputs, labels in dataloaders_new[phase]:
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
                if phase == 'val' and epoch == num_epochs -1:
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
            pred_ = pred_.cpu().numpy()
            true_ = true_.cpu().numpy()
            TP, FP, TN, FN = perf_measure(true_,pred_,pos_label=0)
            print(f"tp: {TP} fp: {FP} tn: {TN} fn: {FN}")
            precision,recall,f_beta_score,support = precision_recall_fscore_support(true_, pred_,pos_label = 0,average='binary')
            f_score_hist.append(f_beta_score)
            print('Precision {:.4f} Recall {:.4f} F_score {:.4f}'.format(precision,recall,f_beta_score))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model,val_acc_history,f_score_hist,time_elapsed

    ######################################################################
    # Visualizing the model predictions
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def visualize_model(model, num_images=16,model_name=''):
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

                outputs = model(inputs)
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
                    ax = plt.subplot(num_images//4, 4, images_so_far)
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
    # model_names = ['alexnet','resnet','squeezenet','densenet']
    num_e = 25


    args = parser.parse_args()
    print(args.model_name)
    model_name_ = args.model_name
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


    model_ft,ohist,fscore_hist,time_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=num_e)
    #####################################################################
    

    visualize_model(model_ft,model_name=model_name_)


    # ######################################################################
    # # ConvNet as fixed feature extractor
    # # ----------------------------------

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
    model_conv,fhist,fscore_hist_f,time_ffe = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=num_e)

    # ######################################################################
    # #
    # Visualize some images
    visualize_model(model_conv,model_name=model_name_)
    # ploting graph
    epochs =num_e
    ohist_ = []
    fscore_hist_ = []
    fscore_hist_f_ = []
    fscore_hist_ = [h for h in fscore_hist]
    fscore_hist_f_ = [h1 for h1 in fscore_hist_f]
    print(time_ft)
    print(time_ffe)
    fig1 = plt.figure(num=model_name_,figsize=(18,9))
    plt.title(f"F_score vs. Number of Training Epochs {model_name_}")
    plt.xlabel("Training Epochs")
    plt.ylabel("F_score")
    plt.plot(range(1,epochs+1),fscore_hist_,label="Fine_tunning")
    plt.plot(range(1,epochs+1),fscore_hist_f_,label="Fixed Feature Extractor")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1,epochs+1,1.0))
    plt.legend()


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
