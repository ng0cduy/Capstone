from commonfunc import *

if __name__ == '__main__':
    source = os.getcwd()
    list_fleece =[]
    for item in os.listdir(source):
        if item.startswith('split'):
            item_path = source + '/' + item
            list_fleece.append(item_path)
    list_fleece = natsort.natsorted(list_fleece)
    split_src_train = []
    split_src_val   = []
    for item in list_fleece:
        for subItem in os.listdir(item):
            subItem_path = item + '/' + subItem
            if subItem == 'train':
                split_src_train.append(subItem_path)
            else:
                split_src_val.append(subItem_path)

    fleeece_data_DUST = source + '/fleece_data_DUST'
    fleeece_data_BURR = source + '/fleece_data_BURR'
    fleeece_data_MBLS = source + '/fleece_data_MBLS'
    fleeece_data_MLKS = source + '/fleece_data_MLKS'
    fleeece_data_MSTN = source + '/fleece_data_MSTN'
    fleeece_data_PCS = source + '/fleece_data_PCS'
    create_folder(fleeece_data_DUST)
    create_folder(fleeece_data_BURR)
    create_folder(fleeece_data_MBLS)
    create_folder(fleeece_data_MLKS)
    create_folder(fleeece_data_MSTN)
    create_folder(fleeece_data_PCS)
    fleece_data_list = [fleeece_data_DUST,fleeece_data_BURR,fleeece_data_MBLS,fleeece_data_MLKS,fleeece_data_MSTN,fleeece_data_PCS]
    # create folder to ignore 1 type of contaminants
    train_path = []
    val_path = [] 
    not_contam_path = []
    contam_path = []
    train_contam =[]
    val_contam =[]
    for fleecedata in fleece_data_list:
        list_ = ['contaminated','not_contaminated']
        for sublist_ in list_:
            sublist_path = fleecedata + '/' + sublist_
            create_folder(sublist_path)
            if sublist_ == 'not_contaminated':
                not_contam_path.append(sublist_path)
            else:
                contam_path.append(sublist_path)
            trainorval =['train','val']
            for _ in trainorval:
                trainorval_path = sublist_path + '/' + _
                create_folder(trainorval_path)
                if sublist_ =='not_contaminated':
                    if _ =='train':
                        train_path.append(trainorval_path)
                    else:
                        val_path.append(trainorval_path)
                else:
                    if _ =='train':
                        train_contam.append(trainorval_path)
                    else:
                        val_contam.append(trainorval_path)
    #copy non_contaminated to each of the folder
    uncontam_path = source +'/not_contaminated'
    uncontam_train = uncontam_path + '/train'
    uncontam_val = uncontam_path + '/val'
    print(train_contam)
    print('\n')
    for item in train_path:
        move_array_folder(src_path=uncontam_train,des=item)
    for itemm in val_path:
        move_array_folder(src_path=uncontam_val,des=itemm)
    
    # print(split_src_val)
    # print('\n')
    # print(split_src_train)
    # print('\n')
    # print(list_fleece)
    # print(fleece_data_list)
    # move train
    list_burr =[]
    list_dust =[]
    list_mbls =[]
    list_mlks =[]
    list_mstn =[]
    list_pcs  =[]
    # for i in list_fleece:
    #     # print(i)
    #     if i.endswith('BURR'):
    #         list_burr.append(i)
    #         for j in fleece_data_list:
    #             if not j.endswith('BURR'):
    #                 list_burr.append(j)
    list_burr = create_list(list_fleece,fleece_data_list,'BURR')
    list_dust = create_list(list_fleece,fleece_data_list,'DUST')
    list_mbls = create_list(list_fleece,fleece_data_list,'MBLS')
    list_mlks = create_list(list_fleece,fleece_data_list,'MLKS')
    list_mstn = create_list(list_fleece,fleece_data_list,'MSTN')
    list_pcs  = create_list(list_fleece,fleece_data_list,'PCS')
    final_list = [list_burr,list_dust,list_dust,list_mbls,list_mlks,list_mstn,list_pcs]
    
    copy_final_list(list=list_burr)
    copy_final_list(list_dust)
    copy_final_list(list_mbls)
    copy_final_list(list_mlks)
    copy_final_list(list_mstn)
    copy_final_list(list_pcs)
    
    