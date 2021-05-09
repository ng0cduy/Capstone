from commonfunc import *

if __name__ == '__main__':
    source = os.getcwd()
    contam_path = source +'/contaminated'
    contam_train = contam_path + '/train'
    contam_val = contam_path + '/val'
    create_folder(contam_path)
    create_folder(contam_train)
    create_folder(contam_val)
    list_ =[]
    for item in os.listdir(source):
        if item.startswith('split'):
            item_path = source + '/' + item
            print(os.path.exists(item_path))
            list_.append(item_path)
    
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
    for fleecedata in fleece_data_list:
        list_ = ['contaminated','not_contaminated']
        for sublist_ in list_:
            sublist_path = fleecedata + '/' + sublist_
            create_folder(sublist_path)
            trainorval =['train','val']
            for _ in trainorval:
                trainorval_path = sublist_path + '/' + _
                create_folder(trainorval_path)

    