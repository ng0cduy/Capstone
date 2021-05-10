'''
this file is used to skip the data. in here will choose 1 frame in each 18 frame
'''
import os
import shutil
import cv2
import re
import natsort
from PIL import Image
def create_folder(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    else:
        shutil.rmtree(path_name)
        os.makedirs(path_name)
    # print(f"{path_name} create successfully")
def produce_bag_list (src):
    list_ = os.listdir(src)
    list_.sort(key=lambda f: int (re.sub('\D','',f)))
    return list_
def process_skip_file (src,des=0,num_of_skip = 18):
    data_list = natsort.natsorted(os.listdir(src))
    for i in range (0,len(data_list),num_of_skip):
        data_path = src + '/' + data_list[i]
        shutil.copy(data_path,des)
        print(data_path)
def check_path_exist(src):
    return os.path.exists(src)
def create_folder_list (array_list=[],des=''):
    for item in array_list:
        item_path = des + '/' + item
        create_folder(item_path)
def produce_bag_list (src):
    list_ = os.listdir(src)
    list_.sort(key=lambda f: int (re.sub('\D','',f)))
    return list_        
def move_array_folder (src_path,des =''):
    '''copy the items in folder array source to destination'''
    for item in os.listdir(src_path):
        item_path = src_path + '/' + item
        shutil.copy(item_path,des)

def create_list (listsend=[],listreceive=[],str=''):
    '''
    create a list for coppying
    liststart: src path

    '''
    res = []
    for j in listreceive:
        if j.endswith(str.upper()):
            res.append(j)
    for item in listsend:
        if not item.endswith(str.upper()):
            res.append(item)
    return res
def copy_final_list (list =[]):
    list_receive = list[0]
    list_receive_contam = list_receive + '/contaminated'
    list_receive_contam_train = list_receive_contam + '/train'
    list_receive_contam_val   = list_receive_contam + '/val'
    if os.path.exists(list_receive_contam_train) and os.path.exists(list_receive_contam_val):
        for i in range(1,len(list),1):
            for trainorval in os.listdir(list[i]):
                trainorval_path = list[i] + '/' + trainorval
                if trainorval =='train':
                    # print('copy from ',trainorval_path,' to',list_receive_contam_train)
                    move_array_folder(src_path=trainorval_path,des=list_receive_contam_train)
                    # pass
                else:
                    # pass
                    # print('copy from ',trainorval_path,' to',list_receive_contam_val)
                    move_array_folder(src_path=trainorval_path,des=list_receive_contam_val)