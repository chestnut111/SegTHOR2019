'''
Related parameter setting of classification model
'''
import pandas as pd 
from easydict import EasyDict as edict
from cls_loss import *
import os 

label_list = ['red','pink','yellow','blue']


config = dict()
#To prepare data 
#Data stored in the root directory, according to their actual situation to adjust

config["BATCH_SIZE"] = 100 
config["gt_class"] = "pink"
config["p_class"] = "p_" + config["gt_class"] 
config["cy"] = 'c'+ config["gt_class"][0] + 'y'
config["cx"] = 'c'+ config["gt_class"][0] + 'x'
config["class_label"] = label_list.index(config["gt_class"]) + 1
config["data_foler"] = '/public/lixin/DATA/segthor/input/'  + "pink-normal/"
config["data_num"] = '3'
config["train_csv_path"] = '/public/lixin/SegTHOR/segthor/pre-process/train_info2.csv'
config["val_csv_path"] = '/public/lixin/SegTHOR/segthor/pre-process/val_info.csv'
config["test_csv_path"] = '/public/lixin/SegTHOR/segthor/pre-process/test_info_add_cls_3.csv' ##yellow


#Recording, training process monitoring
config["monitor"] = 'val_acc'
config["mode"] = 'max'
config["early_p"] = 10
config["reduce_lr_p"] = 5
config["save_weights_only"] = True
config["save_best_only"] = True
config["min_lr"] = 1e-6
config["loss"] = binary_crossentropy



#Parameters of the training phase
config["BACKBONE"] = 'DenseNet121'
config["dropout_rate"] = 0.5
config["NB_EPOCHS"] = 100
config["MAX_TRAIN_STEPS"] = 100
config["encoder_weights"] = 'imagenet'
config["loss"] = focal_loss
config["metrics"] = ['acc']
config["base_model_layer_trainable"] = True
config["pretrain"] = False
config["include_top"] = False

#Keep relevant training record files

config["ex"] = 'try-pink-bce'
config["name"] = '224-val'
config["image_size"] = 112
## other para
config["if_train"] = True
config["pretrain"] = False
config["pretrain_model_path"] = '/public/lixin/SegTHOR/segthor/2d_cls/model_save/DenseNet121/try-yellow-bce/yellow192.h5'
#config["best_model_save_path"]



#4. Keep relevant training record files
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
    
config["model_save_path_notop"]  = os.path.join('model_save/',config["BACKBONE"], config["ex"], config["gt_class"] + config["name"] + 'notop.h5')


config["val_out_csv_name_path"] = os.path.join('model_save/',config["BACKBONE"], config["ex"], config["gt_class"] + config["name"] + '-val-dice-out.csv')

config["val_out_csv_name_path2"] = os.path.join('model_save/',config["BACKBONE"], config["ex"], config["gt_class"] + config["name"] + '-val-pred-out.csv')

config["test_out_csv_name_path"] = os.path.join('model_save/',config["BACKBONE"], config["ex"], config["gt_class"] + config["name"] + '-test-pred-out.csv')

if config["if_train"]:
    df = pd.DataFrame(config)
    df.to_csv(config["conf_save_path"], index = False)

train_config = edict(config)


# print(train_config.class)