# coding: utf-8
import os
import cv2
import random
from random import shuffle
import pandas as pd
import numpy as np
from conf import train_config

from utils import *
from cls_loss import *

#To prepare data
data_foler = train_config.data_foler
data_train = pd.read_csv(train_config.train_csv_path)
data_val = pd.read_csv(train_config.val_csv_path)


# data_yellow = data_train.loc[data_train['yellow']==1]
# ll = [data_yellow, data_yellow.sample(1238), data_train]
# data_train = pd.concat(ll)

ll = [data_train, data_val]
data_all = pd.concat(ll)
from sklearn.model_selection import train_test_split
data_train, data_val = train_test_split(data_all, test_size=0.2, random_state=101010, stratify = data_all['pink'])
 
    
aug_gen = make_image_gen2d_center(data_train, if_aug = True)
VALID_IMG_COUNT = data_val.shape[0]

VALID_IMG_COUNT = len(data_val)
print('The number of val data is {}'.format(VALID_IMG_COUNT))
train_IMG_COUNT = len(data_train)
print('The number of train data is {}'.format(train_IMG_COUNT))

valid_x, valid_y = next(make_image_gen2d_center(data_val, VALID_IMG_COUNT))

                
#Prepare model
import keras
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from classification_models.resnet import ResNet34

model = DenseNet121(weights = train_config.encoder_weights, 
                 include_top= train_config.include_top, 
                 input_shape=(train_config.image_size*2, train_config.image_size*2, 3))
 
from keras.models import Model
def get_model(model,numclasses):
    base_model = model 
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(train_config.dropout_rate)(x) #
    predictions = keras.layers.Dense(numclasses, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    new_model = Model(inputs=base_model.input, outputs=base_model.output)
    # Train top layer
    for layer in base_model.layers:
        layer.trainable = train_config.base_model_layer_trainable
    return model, new_model

#There are two models, one is to save the entire model, and one is not to save the top layer
model, new_model = get_model(model, 1)
from keras.utils import multi_gpu_model ##
muti_model = multi_gpu_model(model, gpus=2) ##



#Network training record
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
         num_train = 10000, #train_IMG_COUNT
         factor=0.5,
         epsilon=0.0001,
         cooldown=2,
         min_lr=train_config.min_lr,
         verbose=1)



#Network training
optimizer = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True) #Adam(0.001) 666
# loss='binary_crossentropy',loss=[focal_loss(alpha=.25, gamma=2)]

muti_model.compile(
              loss='binary_crossentropy', 
              optimizer=keras.optimizers.Adam(0.0001), 
              metrics=train_config.metrics)

MAX_TRAIN_STEPS = train_config.MAX_TRAIN_STEPS
step_count = min(train_config.MAX_TRAIN_STEPS, data_train.shape[0]//train_config.BATCH_SIZE) 

if train_config.if_train:
    loss_history = [muti_model.fit_generator(aug_gen, 
                                 steps_per_epoch=step_count, 
                                 epochs=train_config.NB_EPOCHS, 
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list,
                                workers=1 # the generator is not very thread safe
                                           )]

#Save the single GPU model
from keras.models import load_model
muti_model.load_weights(train_config.best_model_save_path)  
model.save(train_config.model_save_path) 

model.layers.pop()
model.layers.pop()
model.layers.pop()

# model.summary()
new_model.set_weights(model.get_weights())
new_model.save(train_config.model_save_path_notop)

data_val = pd.read_csv(train_config.val_csv_path)
#Offline prediction validation set
cal_val2d_center(data_val, muti_model)

data_test = pd.read_csv(train_config.test_csv_path)
cal_test2d_center(data_test, muti_model)