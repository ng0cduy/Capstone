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
    
    # fleeece_data_DUST = source + '/fleece_data_DUST'
    # fleeece_data_DUST = source + '/fleece_data_DUST'
    # fleeece_data_DUST = source + '/fleece_data_DUST'
    # fleeece_data_DUST = source + '/fleece_data_DUST'
    # fleeece_data_DUST = source + '/fleece_data_DUST'
    # fleeece_data_DUST = source + '/fleece_data_DUST'

    