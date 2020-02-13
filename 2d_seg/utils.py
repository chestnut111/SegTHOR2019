import numpy as np 
import pandas as pd 
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from conf import train_config
from seg_loss import *
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import colorsys
import random
from clr import *

ia.seed(1)
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



def make_image_gen(in_df, batch_size = train_config.BATCH_SIZE, if_aug = False):
    file = in_df['filename'].values
    mask = in_df['maskname'].values
    ys = in_df['ys'].values
    ye = in_df['ye'].values
    xs = in_df['xs'].values
    xe = in_df['xe'].values
    all_batches = np.stack((file,mask, ys, ye, xs, xe),1)
    out_rgb = []
    out_mask = []
    seq = augmentation().to_deterministic()
    while True:
        np.random.shuffle(all_batches)
        for data in all_batches: 
            file = data_folder + data[0]
            mask = data_folder + data[1]
            c_img = crop_pic(cv2.imread(file), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
            c_mask = crop_mask(cv2.imread(mask, 0), ys = data[2],ye = data[3],xs = data[4],xe = data[5])  
            out_rgb += [c_img]
            if if_aug:
                
                out_mask += [ia.SegmentationMapOnImage(c_mask, shape = c_img.shape[:2], nb_classes=1+4)]
            else:
                c_mask = np.where(np.equal(c_mask, train_config.class_label), 1, 0)
                c_mask = np.expand_dims(c_mask, axis=-1)
                out_mask += [c_mask]   
                
            if len(out_rgb)>=batch_size:
                if if_aug:
                    images_aug, segmaps_aug = seq.augment(images=out_rgb, segmentation_maps=out_mask)
                    segmaps_aug = [np.expand_dims(segmaps_aug[i].arr[:,:,train_config.class_label], axis=-1) for i in range(len(segmaps_aug))]
                    
                    out_rgb = np.stack(images_aug, 0)
                    out_mask = segmaps_aug
                else:
                    out_rgb = np.stack(np.array(out_rgb), 0)
                    out_mask = out_mask

                yield out_rgb/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
                
def make_image_gen2d(in_df, batch_size = train_config.BATCH_SIZE, if_aug = False):
    file = in_df['filename'].values
    mask = in_df['maskname'].values
    ys = in_df['ys'].values
    ye = in_df['ye'].values
    xs = in_df['xs'].values
    xe = in_df['xe'].values
    folder = in_df['folder'].values
    all_batches = np.stack((file, mask, ys, ye, xs, xe, folder), 1)
    out_rgb = []
    out_mask = []
    seq = augmentation().to_deterministic()
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
                c_a = crop_mask(cv2.imread(data_folder + after_path, 0), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                c_img = np.stack([c_b,c_in,c_a], -1)
            else:
                file = data_folder + data[0] 
                print(file)
                c_img = crop_pic(cv2.imread(file), ys = data[2],ye = data[3],xs = data[4],xe = data[5])
                
            mask = data_folder + data[1]
            c_mask = crop_mask(cv2.imread(mask, 0), ys = data[2],ye = data[3],xs = data[4],xe = data[5])  
            out_rgb += [c_img]
            if if_aug:
                out_mask += [ia.SegmentationMapOnImage(c_mask, shape = c_img.shape[:2], nb_classes=1+4)]
            else:
                c_mask = np.where(np.equal(c_mask, train_config.class_label), 1, 0)
                c_mask = np.expand_dims(c_mask, axis=-1)
                out_mask += [c_mask]   
                
            if len(out_rgb)>=batch_size:
                if if_aug:
                    images_aug, segmaps_aug = seq.augment(images=out_rgb, segmentation_maps=out_mask)
                    segmaps_aug = [np.expand_dims(segmaps_aug[i].arr[:,:,train_config.class_label], axis=-1) for i in range(len(segmaps_aug))]
                    
                    out_rgb = np.stack(images_aug, 0)
                    out_mask = segmaps_aug
                else:
                    out_rgb = np.stack(np.array(out_rgb), 0)
                    out_mask = out_mask

                yield out_rgb/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []              

def make_image_gen2d_center(in_df, batch_size = train_config.BATCH_SIZE, if_aug = False):
    file = in_df['filename'].values
    mask = in_df['maskname'].values
    y = in_df[train_config.cy].values
    x = in_df[train_config.cx].values
    folder = in_df['folder'].values
    all_batches = np.stack((file, mask, y, x, folder), 1)
    out_rgb = []
    out_mask = []
    seq = augmentation().to_deterministic()
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
#                 print(file,'xxxxxxxxxxx')
                c_img = crop_center_pic(cv2.imread(file), data[2], data[3])
                
            mask = data_folder + data[1]
            c_mask = crop_center_mask(cv2.imread(mask, 0), data[2], data[3])  
            out_rgb += [c_img]
            if if_aug:
                out_mask += [ia.SegmentationMapOnImage(c_mask, shape = c_img.shape[:2], nb_classes=1+4)]
            else:
                c_mask = np.where(np.equal(c_mask, train_config.class_label), 1, 0)
                c_mask = np.expand_dims(c_mask, axis=-1)
                out_mask += [c_mask]   
                
            if len(out_rgb)>=batch_size:
                if if_aug:
                    images_aug, segmaps_aug = seq.augment(images=out_rgb, segmentation_maps=out_mask)
                    segmaps_aug = [np.expand_dims(segmaps_aug[i].arr[:,:,train_config.class_label], axis=-1) for i in range(len(segmaps_aug))]
                    
                    out_rgb = np.stack(images_aug, 0)
                    out_mask = segmaps_aug
                else:
                    out_rgb = np.stack(np.array(out_rgb), 0)
                    out_mask = out_mask

                yield out_rgb/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []                 
                
                
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
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = label_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)
    

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
    max_lr = 0.005 ##
    base_lr = max_lr/10
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
    
    callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]
    return callbacks_list



def cal_val(data_val, model,if_save = False):
    ll = [] #dice for every p 
    p = [] #the index of p 
    for f in data_val['folder'].value_counts().index:
        p.append(f)
        df = data_val.loc[data_val['folder'] == f]
        train_paths = df['filename'].values
        train_mask = df['maskname'].values
        train_cls = df[train_config.gt_class].values #p_class
        train_cls_gt = df[train_config.gt_class].values
        ys = df['ys'].values
        ye = df['ye'].values
        xs = df['xs'].values
        xe = df['xe'].values
        train_all = np.stack((train_paths,train_mask,train_cls,train_cls_gt, ys, ye, xs, xe),1)
        gt = []
        mk = []
        dice = []
        i = 0
        for data in train_all:
            i = i + 1
            c_path = data_folder + data[0]
            c_mask = data_folder + data[1]
            cls_pink = data[2]
            cls_gt = data[3]
            mask = cv2.imread(c_mask, 0)
            mask = np.where(np.equal(mask, train_config.class_label), 1, 0)
            if int(cls_gt) == 1:
                c_img = crop_pic(cv2.imread(c_path),ys = data[4],ye = data[5],xs = data[6],xe = data[7])
                first_img = np.expand_dims(c_img, 0)/255.0
                first_seg = np.round(model.predict(first_img)[0,:,:,0])
                ys = data[4]
                ye = data[5]
                xs = data[6]
                xe = data[7]
                pred = np.zeros((512,512))
                pred[int(np.ceil((ye-100+ys)/2))-train_config.image_size: int(np.ceil((ye-100+ys)/2))+train_config.image_size, int(np.ceil((xe+xs)/2))-train_config.image_size: int(np.ceil((xe+xs)/2))+train_config.image_size] = first_seg
                pred = pred.tolist()
            else:
                pred = np.zeros((512,512)).tolist()
            gt.append(mask.tolist())
            mk.append(pred)
        gt = np.stack(gt, 0)
        mk = np.stack(mk, 0)
        f_fice = dice_index(gt, mk)
        ll.append(f_fice)
        print('The dice of patient {} is {}'.format(f, f_fice))
    print('The mean of dice is {}'.format(np.mean(ll)))
    dic = {'patient':p, 'dice':ll}
    df = pd.DataFrame(dic)
    df.to_csv(train_config.save_dice)
    
import SimpleITK as sitk
from tfmed.utils.med_ops import *

def cal_test(data_test, model, if_save = False):
    ll = [] #dice for every p 
    p = [] #the index of p 
    for f in data_test['folder'].value_counts().index:
        p.append(f)
#         f = 34
        df = data_test.loc[data_test['folder'] == f] 
        path = df['filename'].values[0]
        folder = '/'.join(path.split('/')[:-1])
        print(folder)
        length = len(df)

        mk = []
        data_foler = '/public/lixin/DATA/segthor/input/'
        for i in range(length):
            image = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i) + '.png')

            ys = data_test.loc[data_test['filename'] == image]['ys'].values[0]
            ye = data_test.loc[data_test['filename'] == image]['ye'].values[0]
            xs = data_test.loc[data_test['filename'] == image]['xs'].values[0]
            xe = data_test.loc[data_test['filename'] == image]['xe'].values[0]
            
            c_img = crop_pic(cv2.imread(data_foler + image), ys = ys, ye = ye, xs = xs, xe = xe)
            first_img = np.expand_dims(c_img, 0)/255.0

            pre_cls = data_test.loc[data_test['filename'] == image][train_config.p_class].values[0]

            if pre_cls == 1.0:
                out_red = model.predict(first_img)[0,:,:,0] #np.round()
                pred = np.zeros((512,512))
                pred[int(np.ceil((ye-100+ys)/2))-train_config.image_size: int(np.ceil((ye-100+ys)/2))+train_config.image_size, int(np.ceil((xe+xs)/2))-train_config.image_size: int(np.ceil((xe+xs)/2))+train_config.image_size] = out_red
            else:
                pred = np.zeros((512, 512))
            mk.append(pred.tolist())


        mk = np.stack(mk, 0)
#         mk = np.array(mk,dtype=np.int8)
       
        
        indir = 'segthor_gz/test'
        sd = str(f).zfill(2)
        ori_path = os.path.join(data_foler, indir, 'Patient_' + sd + '.nii.gz') 
        
        o_folder = '/public/lixin/SegTHOR/segthor/post-process/' + train_config.ff
        out_path = os.path.join(o_folder, 'Patient_'+ str(f).zfill(2) +'.nii')
        WriteNiiFromArray(mk, out_path, **GetSettings(ori_path))
        print('done!')
        
def cal_test2d(data_test, model, if_save = False):
    ll = [] #dice for every p 
    p = [] #the index of p 
    for f in data_test['folder'].value_counts().index:
        p.append(f)
#         f = 34
        df = data_test.loc[data_test['folder'] == f] 
        path = df['filename'].values[0]
        folder = '/'.join(path.split('/')[:-1])
        print(folder)
        length = len(df)

        mk = []
        data_foler = '/public/lixin/DATA/segthor/input/'
        for i in range(length):
            image_b = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i-1) + '.png')
            image = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i) + '.png')
            image_a = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i+1) + '.png')

            ys = data_test.loc[data_test['filename'] == image]['ys'].values[0]
            ye = data_test.loc[data_test['filename'] == image]['ye'].values[0]
            xs = data_test.loc[data_test['filename'] == image]['xs'].values[0]
            xe = data_test.loc[data_test['filename'] == image]['xe'].values[0]
            
            if os.path.exists(data_folder + image_b) and os.path.exists(data_folder + image_a):
                c_b = crop_mask(cv2.imread(data_folder + image_b, 0), ys = ys, ye = ye, xs = xs, xe = xe)
                c_in = crop_mask(cv2.imread(data_folder +image, 0), ys = ys, ye = ye, xs = xs, xe = xe)
                c_a = crop_mask(cv2.imread(data_folder + image_a, 0), ys = ys, ye = ye, xs = xs, xe = xe)
                c_img = np.stack([c_b,c_in,c_a], -1)
            else: 
                c_img = crop_pic(cv2.imread(data_folder + image), ys = ys, ye = ye, xs = xs, xe = xe)
                            
#             c_img = crop_pic(cv2.imread(data_foler + image), ys = ys, ye = ye, xs = xs, xe = xe)
            first_img = np.expand_dims(c_img, 0)/255.0

            pre_cls = data_test.loc[data_test['filename'] == image][train_config.p_class].values[0]

            if pre_cls == 1.0:
                out_red = model.predict(first_img)[0,:,:,0] #np.round()
                pred = np.zeros((512,512))
                pred[int(np.ceil((ye-100+ys)/2))-train_config.image_size: int(np.ceil((ye-100+ys)/2))+train_config.image_size, int(np.ceil((xe+xs)/2))-train_config.image_size: int(np.ceil((xe+xs)/2))+train_config.image_size] = out_red
            else:
                pred = np.zeros((512, 512))
            mk.append(pred.tolist())


        mk = np.stack(mk, 0)
#         mk = np.array(mk,dtype=np.int8)
        indir = 'segthor_gz/test'
        sd = str(f).zfill(2)
        ori_path = os.path.join(data_foler, indir, 'Patient_' + sd + '.nii.gz') 
        
        o_folder = '/public/lixin/SegTHOR/segthor/post-process/' + train_config.ff
        if not os.path.isdir(o_folder):
            os.makedirs(o_folder)
        out_path = os.path.join(o_folder, 'Patient_'+ str(f).zfill(2) +'.nii')
        WriteNiiFromArray(mk, out_path, **GetSettings(ori_path))
        print('done!')
        
def cal_test2d_center(data_test, model, if_save = False):
    
    label_file = pd.read_csv('/public/lixin/SegTHOR/segthor/2d_cls/model_save/DenseNet121/try-red-bce/red224-val-test-pred-out.csv')
    center_file = pd.read_csv('/public/lixin/SegTHOR/segthor/post-process/test_2d_center.csv')
    
    print(label_file.head())
    ll = [] #dice for every p 
    p = [] #the index of p 
    for f in data_test['folder'].value_counts().index:
        p.append(f)
#         f = 34
        df = data_test.loc[data_test['folder'] == f] 
        path = df['filename'].values[0]
        folder = '/'.join(path.split('/')[:-1])
        print(folder)
        length = len(df)

        mk = []
        data_foler = '/public/lixin/DATA/segthor/input/' ##????2019-05-11???
        for i in range(length):
            image_b = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i-1) + '.png')
            image = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i) + '.png')
            image_a = os.path.join(folder, 'Patient_' + str(f).zfill(2) +'_' + str(i+1) + '.png')

            pre_cls = data_test.loc[data_test['filename'] == image][train_config.p_class].values[0]
#             pre_cls = label_file.loc[label_file['filename'] == image]['pred_y'].values[0]
            
#             y = center_file.loc[center_file['filename'] == image][train_config.cy].values[0]
#             x = center_file.loc[center_file['filename'] == image][train_config.cx].values[0]

            y = data_test.loc[data_test['filename'] == image][train_config.cy].values[0]
            x = data_test.loc[data_test['filename'] == image][train_config.cx].values[0]    
        
            if pre_cls == 1 and y==1000:
                print(image)
            if  pre_cls==1 and y!=1000: #type(y) != np.float64 and
#                 print(type(y), y)
#                 print(folder)
#                 y = data_test.loc[data_test['filename'] == image][train_config.cy].values[0]
#                 x = data_test.loc[data_test['filename'] == image][train_config.cx].values[0]

                if os.path.exists(data_folder + image_b) and os.path.exists(data_folder + image_a):
                    c_b = crop_center_mask(cv2.imread(data_folder + image_b, 0), y, x)
                    c_in = crop_center_mask(cv2.imread(data_folder +image, 0), y, x)
                    c_a = crop_center_mask(cv2.imread(data_folder + image_a, 0), y, x)
                    c_img = np.stack([c_b,c_in,c_a], -1)
                else: 
                    c_img = crop_center_pic(cv2.imread(data_folder + image), y, x)

                first_img = np.expand_dims(c_img, 0)/255.0
                
                out_red = model.predict(first_img)[0,:,:,0] #np.round()
                pred = np.zeros((512 + train_config.image_size, 512 + train_config.image_size))
                pred[int(y): int(y) + 2*train_config.image_size, int(x) : int(x) + 2*train_config.image_size] = out_red
                pred = pred[train_config.image_size:train_config.image_size+512, train_config.image_size:train_config.image_size+512]
            else:
                pred = np.zeros((512, 512))
            mk.append(pred.tolist())


        mk = np.stack(mk, 0)
#         mk = np.array(mk,dtype=np.int8)
       
        
        indir = 'segthor_gz/test'
        sd = str(f).zfill(2)
        ori_path = os.path.join(data_foler, indir, 'Patient_' + sd + '.nii.gz') 
        
        o_folder = '/public/lixin/SegTHOR/segthor/post-process/' + train_config.ff
        if not os.path.isdir(o_folder):
            os.makedirs(o_folder)
        out_path = os.path.join(o_folder, 'Patient_'+ str(f).zfill(2) +'.nii')
        WriteNiiFromArray(mk, out_path, **GetSettings(ori_path))
        print('done!')        
        

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

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

def cal_val2d(data_val, model,if_save = False):
    ll = [] #dice for every p 
    p = [] #the index of p 
    for f in data_val['folder'].value_counts().index:
        p.append(f)
        df = data_val.loc[data_val['folder'] == f]
        train_paths = df['filename'].values
        train_mask = df['maskname'].values
        train_cls = df[train_config.gt_class].values #p_class
        train_cls_gt = df[train_config.gt_class].values
        ys = df['ys'].values
        ye = df['ye'].values
        xs = df['xs'].values
        xe = df['xe'].values
        train_all = np.stack((train_paths, train_mask, train_cls, train_cls_gt, ys, ye, xs, xe),1)
        gt = []
        mk = []
        dice = []
        i = 0
        for data in train_all:
            i = i + 1
            name_list = data[0].split('_')
            index = int(name_list[-1][:-4])
            
            before_path = '_'.join(name_list[:-1]) + '_'+ str(index-1) + '.png'
            after_path = '_'.join(name_list[:-1]) + '_'+ str(index+1) + '.png'
            
            if os.path.exists(data_folder + before_path) and os.path.exists(data_folder + after_path):
                c_b = crop_mask(cv2.imread(data_folder + before_path, 0), ys = data[4],ye = data[5],xs = data[6],xe = data[7])
                c_in = crop_mask(cv2.imread(data_folder + data[0], 0), ys = data[4],ye = data[5],xs = data[6],xe = data[7])
                c_a = crop_mask(cv2.imread(data_folder + after_path, 0), ys = data[4],ye = data[5],xs = data[6],xe = data[7])
                c_img = np.stack([c_b,c_in,c_a], -1)
            else: 
                c_img = crop_pic(cv2.imread(data_folder + data[0]), ys = data[4],ye = data[5],xs = data[6],xe = data[7])
                
            c_mask = data_folder + data[1]
            cls_pink = data[2]
            cls_gt = data[3]
            mask = cv2.imread(c_mask, 0)
            mask = np.where(np.equal(mask, train_config.class_label), 1, 0)
            if int(cls_gt) == 1:
#                 c_img = crop_pic(cv2.imread(c_path),ys = data[4],ye = data[5],xs = data[6],xe = data[7])
                first_img = np.expand_dims(c_img, 0)/255.0
                first_seg = np.round(model.predict(first_img)[0,:,:,0])
                ys = data[4]
                ye = data[5]
                xs = data[6]
                xe = data[7]
                pred = np.zeros((512,512))
                pred[int(np.ceil((ye-100+ys)/2))- train_config.image_size: int(np.ceil((ye-100+ys)/2))+train_config.image_size, int(np.ceil((xe+xs)/2))-train_config.image_size: int(np.ceil((xe+xs)/2))+train_config.image_size] = first_seg
                pred = pred.tolist()
            else:
                pred = np.zeros((512,512)).tolist()
            gt.append(mask.tolist())
            mk.append(pred)
        gt = np.stack(gt, 0)
        mk = np.stack(mk, 0)
        f_fice = dice_index(gt, mk)
        ll.append(f_fice)
        print('The dice of patient {} is {}'.format(f, f_fice))
    print('The mean of dice is {}'.format(np.mean(ll)))
    dic = {'patient':p, 'dice':ll}
    df = pd.DataFrame(dic)
    df.to_csv(train_config.save_dice)
    
def cal_val2d_center(data_val, model,if_save = False):
    ll = [] #dice for every p 
    p = [] #the index of p 
#     label_file = pd.read_csv('/public/lixin/SegTHOR/segthor/2d_cls/model_save/DenseNet121/try-red-bce/red96-val-val-pred-out.csv')
    for f in data_val['folder'].value_counts().index:
        p.append(f)
        df = data_val.loc[data_val['folder'] == f]
        train_paths = df['filename'].values
        train_mask = df['maskname'].values
        train_cls = df[train_config.gt_class].values #p_class
        train_cls_gt = df[train_config.gt_class].values
        y = df[train_config.cy].values  ##
        x = df[train_config.cx].values  ##
        train_all = np.stack((train_paths, train_mask, train_cls, train_cls_gt, y, x), 1)
        gt = []
        mk = []
        dice = []
        i = 0
        for data in train_all:
            i = i + 1
            name_list = data[0].split('_')
            index = int(name_list[-1][:-4])
            
            before_path = '_'.join(name_list[:-1]) + '_'+ str(index-1) + '.png'
            after_path = '_'.join(name_list[:-1]) + '_'+ str(index+1) + '.png'
                
            c_mask = data_folder + data[1]
            cls_pink = data[2]
            cls_gt = data[3]
            mask = cv2.imread(c_mask, 0)
            mask = np.where(np.equal(mask, train_config.class_label), 1, 0)
            
                       
#             pred_cls = label_file.loc[label_file['filename'] == data[0]]['p_yellow'].values[0] ##2019-05-08
            
            if int(cls_gt) == 1:
                
                if os.path.exists(data_folder + before_path) and os.path.exists(data_folder + after_path):
                    c_b = crop_center_mask(cv2.imread(data_folder + before_path, 0), data[4], data[5])
                    c_in = crop_center_mask(cv2.imread(data_folder + data[0], 0), data[4],data[5])
                    c_a = crop_center_mask(cv2.imread(data_folder + after_path, 0), data[4], data[5])
                    c_img = np.stack([c_b,c_in,c_a], -1)
                else: 
                    c_img = crop_center_pic(cv2.imread(data_folder + data[0]), data[4],data[5])
                first_img = np.expand_dims(c_img, 0)/255.0
                first_seg = np.round(model.predict(first_img)[0,:,:,0])
                y = int(data[4])
                x = int(data[5])
#                 pred = np.zeros((512 + 2*train_config.image_size, 512 + 2*train_config.image_size))  ###？？
                pred = np.pad(np.zeros((512,512)), ((train_config.image_size,train_config.image_size),(train_config.image_size,train_config.image_size)),'constant' ) #2019-05-13
                
                pred[y: y + 2*train_config.image_size, x : x + 2*train_config.image_size] = first_seg
                
                pred = pred[train_config.image_size:train_config.image_size+512, train_config.image_size:train_config.image_size+512].tolist()  ##
                
#                 pred = first_seg                
            else:
                pred = np.zeros((512,512)).tolist()
#                 pred = np.zeros((2*train_config.image_size,2*train_config.image_size)).tolist()
    
#             mask = crop_center_mask(mask, data[4], data[5])

            gt.append(mask.tolist())
            mk.append(pred)
            
        gt = np.stack(gt, 0)
        mk = np.stack(mk, 0)
        f_fice = dice_index(gt, mk)
        ll.append(f_fice)
        print('The dice of patient {} is {}'.format(f, f_fice))
    print('The mean of dice is {}'.format(np.mean(ll)))
    dic = {'patient':p, 'dice':ll}
    df = pd.DataFrame(dic)
    df.to_csv(train_config.save_dice)