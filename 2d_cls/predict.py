# coding: utf-8
import os
import cv2
import random
from random import shuffle
import pandas as pd
import numpy as np
from conf import train_config

from utils import *
from loss import *

#To prepare data
data_foler = train_config.data_foler
data_train = pd.read_csv(train_config.train_csv_path)
data_val = pd.read_csv(train_config.val_csv_path)


#Prepare model
import keras
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from classification_models.resnet import ResNet34

model = ResNet34(weights = train_config.encoder_weights, 
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
    # Train top layer
    for layer in base_model.layers:
        layer.trainable = train_config.base_model_layer_trainable
    return model

#There are two models, one is to save the entire model, and one is not to save the top layer
model = get_model(model, 1)
model.load_weights(train_config.model_save_path) 
from keras.utils import multi_gpu_model 
muti_model = multi_gpu_model(model, gpus=2) 

def cal_val(df, model):
    
    pred = []
    train_paths = df['filename'].values
    train_cls_gt = df[train_config.gt_class].values

    ys = df['ys'].values
    ye = df['ye'].values
    xs = df['xs'].values
    xe = df['xe'].values

    train_all = np.stack((train_paths, train_cls_gt, ys, ye, xs, xe), 1)

    for data in train_all:
        c_path = data_foler + data[0]
        cls_gt = data[1]
        c_img = crop_pic(cv2.imread(c_path), ys = data[2], ye = data[3], xs = data[4], xe = data[5])
        first_img = np.expand_dims(c_img, 0)/255.0
        temp = ((model.predict(first_img)[0].ravel()*model.predict(first_img[:, ::-1, :, :])[0].ravel()*model.predict(first_img[:, ::-1, ::-1, :])[0].ravel()*model.predict(first_img[:, :, ::-1, :])[0].ravel())**0.25)
        pred_out = np.round(temp)[0]
        
        pred.append(pred_out)
        
    return pred
pred = cal_val(data_val, muti_model)

print(len(pred))
data_val[train_config.p_class] = pred
data_val.to_csv('train_csv/val_cls.csv', index=False)

