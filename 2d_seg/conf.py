'''
Related parameter Settings of segmentation model
'''
from easydict import EasyDict as edict
from seg_loss import  *
import os 
import pandas as pd 

#There are four segmentation annotations
label_list = ['red','pink','yellow','blue']
#Establishing a parameter dictionary
config = dict()
#1. To prepare data 
config["data_foler"] = '/public/lixin/DATA/segthor/input/' + 'pink-normal/'#Data stored in the root directory, according to their actual situation to adjust
config["gt_class"] = "pink" #Select a category  #red folder is median data#

config["cy"] = 'c'+ config["gt_class"][0] + 'y' #
config["cx"] = 'c'+ config["gt_class"][0] + 'x' #

config["p_class"] = 'p_' + config["gt_class"]
config["class_label"] = label_list.index(config["gt_class"]) + 1
config["data_num"] = '3' #Select the data folder to train
config["train_csv_path"] = '/public/lixin/SegTHOR/segthor/pre-process/train_info2.csv'
config["val_csv_path"] = '/public/lixin/SegTHOR/segthor/pre-process/val_info.csv'
config["test_csv_path"] = '/public/lixin/SegTHOR/segthor/pre-process/test_info_add_cls_3.csv' ##yellow

#2. Relevant parameters during training
config["BATCH_SIZE"] = 60
config["BACKBONE"] = 'senet154' ##
config["NB_EPOCHS"] = 100
config["MAX_TRAIN_STEPS"] = 100
config["base_lr"] = 1e-3
config["weight_decay"] = 1e-6
config["momentum"] = 0.9
config["metrics"] = [dice_coef, 'binary_accuracy', true_positive_rate]
config["loss"] = dice_p_bce

#dice_p_bce 

#3. Recording, training process monitoring
config["monitor"] = 'val_dice_coef'
config["mode"] = 'max'
config["early_p"] = 8
config["reduce_lr_p"] = 4
config["save_weights_only"] = True
config["save_best_only"] = True
config["min_lr"] = 1e-6
config["encoder_weights"] = 'imagenet'



#4. Keep relevant training record files
config["ex"] = 'try-pink' ##
config["name"] = '224'
path = os.path.join('model_save/', config["BACKBONE"], config["ex"])
if not os.path.isdir(path):
    os.makedirs(path)
    
config["log_csv_name"] = config["gt_class"] + config["name"] + '.csv'
config["log_csv_path"] = os.path.join('model_save/',config["BACKBONE"], config["ex"], config["log_csv_name"])
config["best_model_save_name"] = '{}_weights.best.hdf5'.format('pretrain_model')
config["best_model_save_path"] = os.path.join('model_save/',config["BACKBONE"], config["ex"], config["best_model_save_name"])
config["model_save_name"] = config["gt_class"] + config["name"] + '.h5'
config["model_save_path"] = os.path.join('model_save/',config["BACKBONE"], config["ex"], config["model_save_name"])
config["val_dice_file"] = config["gt_class"] + config["name"]+ '_val_dice.csv'
config["save_dice"] = os.path.join('model_save/' , config["BACKBONE"], config["ex"], config["val_dice_file"])

config["conf_save"] = 'config'+ config["name"] + '.csv'
config["conf_save_path"] = os.path.join('model_save/' , config["BACKBONE"],  config["ex"], config["conf_save"])

config["val_out_csv_name"] = 'val_cal' + config["name"] + '.csv'
config["val_out_csv_name_path"] = os.path.join('model_save/' , config["BACKBONE"],  config["ex"], config["val_out_csv_name"])


## other para
config["if_train"] = True
config["pretrain"] = False

config["image_size"] =  112 #half of pic size
config["ff"] = '44' 
config["pretrain_model_path"] = '/public/lixin/SegTHOR/segthor/2d_seg/model_save/densenet121/try224-red-normal-all-use-cls-para/red160.h5'
#config["best_model_save_path"]

if config["if_train"]:
    df = pd.DataFrame(config)
    df.to_csv(config["conf_save_path"], index = False)

train_config = edict(config)

# print(train_config.class_label)