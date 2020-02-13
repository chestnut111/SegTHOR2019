# coding: utf-8
import numpy as np 
import pandas as pd 
import cv2
import os 
import matplotlib.pyplot as plt
import random

# from utils import show_loss
from seg_loss import  *
from utils import *
from conf import train_config

data_foler = train_config.data_foler

# To prepare data
data_train = pd.read_csv(train_config.train_csv_path)
data_val = pd.read_csv(train_config.val_csv_path)
data_train_mb = data_train.loc[data_train[train_config.gt_class] == 1]
data_val_mb = data_val.loc[data_val[train_config.gt_class] == 1]

# data_train_mb = data_train ###
# data_val_mb = data_val ###

###
ll = [data_train_mb, data_val_mb]
data_all = pd.concat(ll)
from sklearn.model_selection import train_test_split
data_train_mb, data_val_mb = train_test_split(data_all, test_size=0.2, random_state=101010, stratify = data_all['red'])
###


VALID_IMG_COUNT = len(data_val_mb)
print('The number of val data is {}'.format(VALID_IMG_COUNT))
train_IMG_COUNT = len(data_train_mb)
print('The number of train data is {}'.format(train_IMG_COUNT))

valid_x, valid_y = next(make_image_gen2d_center(data_val_mb, VALID_IMG_COUNT))
aug_gen = make_image_gen2d_center(data_train_mb, if_aug = True) #use imgaug s


## Training parameter preparation for call_bask
from utils import call
callbacks_list = call(
         weight_path = train_config.best_model_save_path, 
         monitor = train_config.monitor , 
         mode = train_config.mode, 
         reduce_lr_p = train_config.reduce_lr_p, 
         early_p = train_config.early_p, 
         log_csv_path = train_config.log_csv_path,
         save_best_only = train_config.save_best_only, 
         save_weights_only = train_config.save_weights_only,
         num_train = 5000, #train_IMG_COUNT
         factor=0.5,
         epsilon=0.0001,
         cooldown=2,
         min_lr=train_config.min_lr,
         verbose=1)

from segmentation_models import Unet
model = Unet(train_config.BACKBONE, encoder_weights = train_config.encoder_weights, encoder_freeze=False)
step_count = min(train_config.MAX_TRAIN_STEPS, 5000//train_config.BATCH_SIZE)

if train_config.pretrain:
    model.load_weights(train_config.pretrain_model_path)

from keras.utils import multi_gpu_model 
muti_model = multi_gpu_model(model, gpus=4) 
muti_model.compile(optimizer = keras.optimizers.Adam(0.001), loss = train_config.loss, metrics=train_config.metrics)

if train_config.if_train:
    loss_history = [muti_model.fit_generator(aug_gen, 
                            steps_per_epoch=step_count, 
                            epochs=train_config.NB_EPOCHS, 
                            validation_data=(valid_x, valid_y),
                            callbacks=callbacks_list,
                           workers=1 
                                      )]
    
# muti_model.load_weights(train_config.best_model_save_path)  # save single gpu model 
# model.save(train_config.model_save_path)
    
# show_loss(loss_history)

data_val = pd.read_csv(train_config.val_csv_path) ##
cal_val2d_center(data_val, muti_model)

# cal val dice for every patient
data_test = pd.read_csv(train_config.test_csv_path)
cal_test2d_center(data_test, muti_model, if_save = True)


