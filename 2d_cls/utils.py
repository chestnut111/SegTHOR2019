import numpy as np
import pandas as pd 
import cv2
import os
import matplotlib.pyplot as plt
import itertools
from keras.callbacks import  ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint

from conf import train_config
from cls_loss import *

from clr import *


data_folder = train_config.data_foler

def crop_pic(pic, ys, ye, xs, xe):
    return pic[int(np.ceil((ye-100+ys)/2))-train_config.image_size: int(np.ceil((ye-100+ys)/2))+train_config.image_size, int(np.ceil((xe+xs)/2))-train_config.image_size: int(np.ceil((xe+xs)/2))+train_config.image_size,:]

def crop_mask(pic, ys, ye, xs, xe):
    return pic[int(np.ceil((ye-100+ys)/2))-train_config.image_size: int(np.ceil((ye-100+ys)/2))+train_config.image_size, int(np.ceil((xe+xs)/2))-train_config.image_size: int(np.ceil((xe+xs)/2))+train_config.image_size]

def crop_center_pic(pic, y, x):
    pic = np.pad(pic, ((train_config.image_size,train_config.image_size),(train_config.image_size,train_config.image_size),(0,0)),'constant' )
    return pic[int(y): int(y) + 2*train_config.image_size, int(x): int(x) + 2*train_config.image_size,:]

def crop_center_mask(pic, y, x):
    pic = np.pad(pic, ((train_config.image_size,train_config.image_size),(train_config.image_size,train_config.image_size)),'constant' ) 
    return  pic[int(y): int(y) + 2*train_config.image_size, int(x): int(x) + 2*train_config.image_size]


def make_image_gen2d(in_df, batch_size = train_config.BATCH_SIZE, if_aug =  False):
    file = in_df['filename'].values
    label = in_df[train_config.gt_class].values

    ys = in_df['ys'].values
    ye = in_df['ye'].values
    xs = in_df['xs'].values
    xe = in_df['xe'].values
    
    all_batches = np.stack((file, label, ys, ye, xs, xe),1)
    out_rgb = []
    out_y = []
    seq = augmentation()
    while True:
        np.random.shuffle(all_batches)
        for data in all_batches: 
            name_list = data[0].split('_')
            index = int(name_list[-1][:-4])
            
            before_path = '_'.join(name_list[:-1]) + '_'+ str(index-1) + '.png'
            after_path = '_'.join(name_list[:-1]) + '_'+ str(index+1) + '.png'
            
            if os.path.exists(data_folder + before_path) and os.path.exists(data_folder + after_path):
                c_b = crop_mask(cv2.imread(data_folder + before_path, 0), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                c_in = crop_mask(cv2.imread(data_folder + data[0] , 0), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                c_a = crop_mask(cv2.imread(data_folder + after_path, 0),ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                c_img = np.stack([c_b,c_in,c_a], -1)
            else:
                file = data_folder + data[0] 
                c_img = crop_pic(cv2.imread(file), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                
            out_rgb += [c_img]
            out_y.append([data[1]]) 
            
            if len(out_rgb) >= batch_size:
                if if_aug:
                    out_rgb = seq.augment_images(out_rgb)
                    
                yield np.stack(out_rgb, 0)/255.0, np.array(out_y)
                out_rgb, out_y = [], []
                
                
def make_image_gen2d_center(in_df, batch_size = train_config.BATCH_SIZE, if_aug = False):
    file = in_df['filename'].values
    label = in_df[train_config.gt_class].values
    
    y = in_df[train_config.cy].values # red cry
    x = in_df[train_config.cx].values # red crx
    folder = in_df['folder'].values
    
    all_batches = np.stack((file, label, y, x), 1)
    out_rgb = []
    out_y = []
    
    seq = augmentation()
    while True:
        np.random.shuffle(all_batches)
        for data in all_batches: 
            
            name_list = data[0].split('_')
            index = int(name_list[-1][:-4])
            
            before_path = '_'.join(name_list[:-1]) + '_'+ str(index-1) + '.png'
            after_path = '_'.join(name_list[:-1]) + '_'+ str(index+1) + '.png'
            
            if os.path.exists(data_folder + before_path) and os.path.exists(data_folder + after_path):
                c_b = crop_center_mask(cv2.imread(data_folder + before_path, 0), data[2], data[3])
                c_in = crop_center_mask(cv2.imread(data_folder + data[0] , 0), data[2], data[3])
                c_a = crop_center_mask(cv2.imread(data_folder + after_path, 0), data[2], data[3])
                c_img = np.stack([c_b,c_in,c_a], -1)
            else:
                file = data_folder + data[0] 
                c_img = crop_center_pic(cv2.imread(file), data[2], data[3])
                
            out_rgb += [c_img]
            out_y.append([data[1]])    
                
            if len(out_rgb) >= batch_size:
                if if_aug:
                    out_rgb = seq.augment_images(out_rgb)
                    
                yield np.stack(out_rgb, 0)/255.0, np.array(out_y)
                out_rgb, out_y = [], []               
                    

def call(weight_path, 
         monitor, mode, 
         reduce_lr_p, 
         early_p, 
         log_csv_path,
         save_best_only = True, 
         save_weights_only = True,
         num_train = 0,
         factor=0.5,
         epsilon=0.0001,
         cooldown=2,
         min_lr=1e-6,
         verbose=1):
    
    batch_size = train_config.BATCH_SIZE
    epochs = train_config.NB_EPOCHS
    max_lr = 0.0001 ##
    base_lr = max_lr/20
    max_m = 0.98
    base_m = 0.85

    cyclical_momentum = False
    augment = True
    cycles = 2.35

    iterations = round(num_train/batch_size*epochs)
    iterations = list(range(0, iterations+1))
    step_size = len(iterations)/(cycles)


    clr =  OneCyclicLR(base_lr=base_lr,
                    max_lr=max_lr,
                    step_size=step_size,
                    max_m=max_m,
                    base_m=base_m,
                    cyclical_momentum=cyclical_momentum)
    
#     weight_path = train_config.best_model_save_path
    csv_logger = CSVLogger(log_csv_path, append=True, separator=',')
    checkpoint = ModelCheckpoint(weight_path, monitor = monitor, verbose=verbose, 
                                 save_best_only=save_best_only, mode=mode, save_weights_only = save_weights_only)

    reduceLROnPlat = ReduceLROnPlateau(monitor=monitor, factor=factor, 
                                       patience = reduce_lr_p, 
                                       verbose=1, mode = mode, epsilon=epsilon, cooldown=cooldown, min_lr=min_lr)

    early = EarlyStopping(monitor=monitor, 
                          mode=mode, 
                          patience= early_p) # probably needs to be more patient
    
    callbacks_list = [checkpoint, early,  csv_logger, reduceLROnPlat]
    return callbacks_list


   
def cal_val(data_val, model):
    folder = []
    tp = []
    fn = []
    fp = []
    tn = []
    for f in data_val['folder'].value_counts().index:
        df = data_val.loc[data_val['folder'] == f]
        train_paths = df['filename'].values
        train_cls_gt = df[train_config.gt_class].values

        ys = df['ys'].values
        ye = df['ye'].values
        xs = df['xs'].values
        xe = df['xe'].values

        train_all = np.stack((train_paths, train_cls_gt, ys, ye, xs, xe), 1)

        tpp = 0 #The number of true positive samples
        fnn = 0 #The number of false negative samples
        
        fpp = 0 #The number of false positive samples
        tnn = 0 #The number of true negative samples
        
        for data in train_all:
            c_path = data_foler + data[0]
            cls_gt = data[1]
            c_img = crop_pic(cv2.imread(c_path), ys = data[2], ye = data[3], xs = data[4], xe = data[5])
            first_img = np.expand_dims(c_img, 0)/255.0
            temp = ((model.predict(first_img)[0].ravel()*model.predict(first_img[:, ::-1, :, :])[0].ravel()*model.predict(first_img[:, ::-1, ::-1, :])[0].ravel()*model.predict(first_img[:, :, ::-1, :])[0].ravel())**0.25)
            pred_out = np.round(temp)[0]
            if pred_out == cls_gt and cls_gt == 1:
                tpp += 1
            if pred_out != cls_gt and cls_gt == 1:
                fnn += 1
            if pred_out != cls_gt and cls_gt == 0:
                fpp += 1
            if pred_out == cls_gt and cls_gt == 0:
                tnn += 1
                
        folder.append(f)
        tp.append(tpp)
        fn.append(fnn)
        fp.append(fpp)
        tn.append(tnn)
        
        print('The patient of {} , tp is {}, fn is {}, fp is {}, tn is {}'.format(f, tpp, fnn, fpp, tnn))
        
    dic = {'folder':folder, 
          'tp':tp,
          'fn':fn,
          'fp':fp,
          'tn':tn}
    df = pd.DataFrame(dic)
    df.to_csv(train_config.val_out_csv_name_path, index=False)

    

def cal_val2d(data_val, model):
    folder = []
    tp = []
    fn = []
    fp = []
    tn = []
    for f in data_val['folder'].value_counts().index:
        df = data_val.loc[data_val['folder'] == f]
        train_paths = df['filename'].values
        train_cls_gt = df[train_config.gt_class].values

        ys = df['ys'].values
        ye = df['ye'].values
        xs = df['xs'].values
        xe = df['xe'].values

        train_all = np.stack((train_paths, train_cls_gt, ys, ye, xs, xe), 1)

        tpp = 0 #The number of true positive samples
        fnn = 0 #The number of false negative samples
        
        fpp = 0 #The number of false positive samples
        tnn = 0 #The number of true negative samples
        
        for data in train_all:
            name_list = data[0].split('_')
            index = int(name_list[-1][:-4])
            
            before_path = '_'.join(name_list[:-1]) + '_'+ str(index-1) + '.png'
            after_path = '_'.join(name_list[:-1]) + '_'+ str(index+1) + '.png'
            
            if os.path.exists(data_folder + before_path) and os.path.exists(data_folder + after_path):
                c_b = crop_mask(cv2.imread(data_folder + before_path, 0), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                c_in = crop_mask(cv2.imread(data_folder + data[0] , 0), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                c_a = crop_mask(cv2.imread(data_folder + after_path, 0),ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                c_img = np.stack([c_b,c_in,c_a], -1)
            else:
                file = data_folder + data[0] 
                c_img = crop_pic(cv2.imread(file), ys = data[2],ye = data[3],xs = data[4],xe = data[5]) 

            first_img = np.expand_dims(c_img, 0)/255.0
            temp = ((model.predict(first_img)[0].ravel()*model.predict(first_img[:, ::-1, :, :])[0].ravel()*model.predict(first_img[:, ::-1, ::-1, :])[0].ravel()*model.predict(first_img[:, :, ::-1, :])[0].ravel())**0.25)
            pred_out = np.round(temp)[0]
            cls_gt = data[1]
            if pred_out == cls_gt and cls_gt == 1:
                tpp += 1
            if pred_out != cls_gt and cls_gt == 1:
                fnn += 1
            if pred_out != cls_gt and cls_gt == 0:
                fpp += 1
            if pred_out == cls_gt and cls_gt == 0:
                tnn += 1
                
        folder.append(f)
        tp.append(tpp)
        fn.append(fnn)
        fp.append(fpp)
        tn.append(tnn)
        
        print('The patient of {} , tp is {}, fn is {}, fp is {}, tn is {}'.format(f, tpp, fnn, fpp, tnn))
        
    dic = {'folder':folder, 
          'tp':tp,
          'fn':fn,
          'fp':fp,
          'tn':tn}
    df = pd.DataFrame(dic)
    df.to_csv(train_config.val_out_csv_name_path, index=False)
    
def cal_val2d_center(data_val, model):
    folder = []
    tp = []
    fn = []
    fp = []
    tn = []
    
    pic = []
    pred = []
    gt = []
    
    
    for f in data_val['folder'].value_counts().index: #data_val['folder'].value_counts().index
        df = data_val.loc[data_val['folder'] == f]
        train_paths = df['filename'].values
        train_cls_gt = df[train_config.gt_class].values
        train_cls = df[train_config.gt_class].values #p_class
        

        y = df[train_config.cy].values #
        x = df[train_config.cx].values #

        train_all = np.stack((train_paths, train_cls_gt, train_cls, y, x), 1)

        tpp = 0 #The number of true positive samples
        fnn = 0 #The number of false negative samples
        
        fpp = 0 #The number of false positive samples
        tnn = 0 #The number of true negative samples
        
        for data in train_all:
            name_list = data[0].split('_')
            index = int(name_list[-1][:-4])
            
            before_path = '_'.join(name_list[:-1]) + '_'+ str(index-1) + '.png'
            after_path = '_'.join(name_list[:-1]) + '_'+ str(index+1) + '.png'
            
            if os.path.exists(data_folder + before_path) and os.path.exists(data_folder + after_path):
                c_b = crop_center_mask(cv2.imread(data_folder + before_path, 0), y = data[3], x = data[4])
                c_in = crop_center_mask(cv2.imread(data_folder + data[0] , 0), y = data[3], x = data[4])
                c_a = crop_center_mask(cv2.imread(data_folder + after_path, 0), y = data[3], x = data[4])
                c_img = np.stack([c_b,c_in,c_a], -1)
            else:
                file = data_folder + data[0] 
                c_img = crop_center_pic(cv2.imread(file), y = data[3], x = data[4]) 

            first_img = np.expand_dims(c_img, 0)/255.0
            temp = ((model.predict(first_img)[0].ravel()*model.predict(first_img[:, ::-1, :, :])[0].ravel()*model.predict(first_img[:, ::-1, ::-1, :])[0].ravel()*model.predict(first_img[:, :, ::-1, :])[0].ravel())**0.25)
            pred_out = np.round(temp)[0]
            cls_gt = data[1]
            if pred_out == cls_gt and cls_gt == 1:
                tpp += 1
            if pred_out != cls_gt and cls_gt == 1:
                fnn += 1
                if f == 3:
                    print(data[0])
            if pred_out != cls_gt and cls_gt == 0:
                fpp += 1
            if pred_out == cls_gt and cls_gt == 0:
                tnn += 1
            
            pic.append(data[0])
            pred.append(pred_out)
            gt.append(data[1])
            
                
        folder.append(f)
        tp.append(tpp)
        fn.append(fnn)
        fp.append(fpp)
        tn.append(tnn)
        
        print('The patient of {} , tp is {}, fn is {}, fp is {}, tn is {}'.format(f, tpp, fnn, fpp, tnn))
        
    dic = {'folder':folder, 
          'tp':tp,
          'fn':fn,
          'fp':fp,
          'tn':tn}
    df = pd.DataFrame(dic)
    df.to_csv(train_config.val_out_csv_name_path, index=False)   
    
    dic2 = {'filename': pic,
           'p_yellow' : pred,
           'gt':gt}
    
    df2 = pd.DataFrame(dic2)
    df2.to_csv(train_config.val_out_csv_name_path2, index=False)

def cal_test2d_center(data_test, model, if_save = False):
    ll = [] #
    p = [] #the index of p 
    
    pic = []
    mk = []
    
    for f in data_test['folder'].value_counts().index:
        p.append(f)
#         f = 34
        df = data_test.loc[data_test['folder'] == f] 
        path = df['filename'].values[0]
        folder = '/'.join(path.split('/')[:-1])
        print(folder)
        length = len(df)

        
        data_foler = train_config.data_foler
        for i in range(length):
            image_b = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i-1) + '.png')
            image = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i) + '.png')
            image_a = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i+1) + '.png')

            y = data_test.loc[data_test['filename'] == image][train_config.cy].values[0]  #
            x = data_test.loc[data_test['filename'] == image][train_config.cx].values[0]  #

            if os.path.exists(data_folder + image_b) and os.path.exists(data_folder + image_a):
                c_b = crop_center_mask(cv2.imread(data_folder + image_b, 0), y, x)
                c_in = crop_center_mask(cv2.imread(data_folder +image, 0), y, x)
                c_a = crop_center_mask(cv2.imread(data_folder + image_a, 0), y, x)
                c_img = np.stack([c_b,c_in,c_a], -1)
            else: 
                c_img = crop_center_pic(cv2.imread(data_folder + image), y, x)
                            
            first_img = np.expand_dims(c_img, 0)/255.0
            temp = ((model.predict(first_img)[0].ravel()*model.predict(first_img[:, ::-1, :, :])[0].ravel()*model.predict(first_img[:, ::-1, ::-1, :])[0].ravel()*model.predict(first_img[:, :, ::-1, :])[0].ravel())**0.25)
            pred_out = np.round(temp)[0]
            
            pic.append(image)
            mk.append(pred_out)
    print(len(pic))
    print(len(mk))
    dic = {'filename': pic,
          'pred_y': mk}
    df = pd.DataFrame(dic)
    
    df.to_csv(train_config.test_out_csv_name_path, index=False)
            
                


     
        
    
    

# Augmentation
from imgaug import augmenters as iaa
import imgaug as ia

def augmentation():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            
            # crop images by -10% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.1, 0.1),
                pad_mode=ia.ALL,
                pad_cval=0)),
            
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=0, # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            
            iaa.SomeOf((0, 4),
                [    
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # add gaussian noise to images
                    iaa.AddToHueAndSaturation((-1, 1)) # change hue and saturation
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq


#Data enhancement and data acquisition
from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.01, 
                  height_shift_range = 0.01, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True, ###
                  fill_mode = "constant",
                   cval = 0,
                   data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen):
    for in_x, in_y in in_gen:
        a = image_gen.flow(255*in_x,in_y,batch_size = in_x.shape[0])
        x, y = next(a)
        yield x/255.0, y
        
def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def maxAreaOfIsland(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    m, n = len(grid), len(grid[0])

    areas = []
    for i in range(m):
        for j in range(n):
            if grid[i][j]:
                areas.append(cal_area(grid, i, j))
    return max(areas) if areas else 0

def cal_area(grid, i, j):

    m, n = len(grid), len(grid[0])
    count = 1
    grid[i][j] = 0
    queue = [(i,j)]

    if len(queue) != 0:
        o,p = queue.pop(0)
        # up
        if 0 < o and grid[o-1][p]:
            grid[o-1][p] = 0
            count += 1
            queue.append((o-1, p))

        #down
        if o+1 < m and grid[o+1][p]:
            grid[o+1][p] = 0
            count += 1
            queue.append((o+1, p))

        #left
        if 0 < p and grid[o][p-1]:
            grid[o][p-1] = 0
            count += 1
            queue.append((o, p-1))

        #right
        if p+1 < n and grid[o][p+1]:
            grid[o][p+1] = 0
            count += 1
            queue.append((o, p+1))

    return count
        