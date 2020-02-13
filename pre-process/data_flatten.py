import os
import argparse
import nibabel as nib
import glob
import cv2
import numpy as np
import random
import SimpleITK as sitk
import pandas as pd

 
'''
Automatic window width and window position 
adjustment requires the average pixel value 
of each organ as the window position, 
and some experiments need to be tested
'''

media_csv_train = pd.read_csv('blue-train_media.csv')
media_csv_val = pd.read_csv('blue-val_media.csv')

media_csv_test = pd.read_csv('blue-test_media.csv')

ll = [media_csv_train, media_csv_val, media_csv_test ]
mean_csv = pd.concat(ll)

'''
Set global variables to save image information to a CSV file
'''
data_path = []
mask_path = []
red = []
pink = []
yellow = []
blue = []
ys = []
ye = []
xs = []
xe = []
folder = []

#Data placement folder
data_foler = '/public/lixin/DATA/segthor/input/'

def read_image(input_nii, mean = 50, scale = False):
    nii = sitk.ReadImage(input_nii)
    data = sitk.GetArrayFromImage(nii)
    crop = get_crop_index(data)
    if scale:
        mi = -125 + (mean - 50)
        ma = 225 + (mean - 50)
        data = np.where(data > ma, ma, data)
        data = np.where(data < mi, mi, data)
    data = data.astype(np.int16)
    data = (data - data.min()) / (data.max() - data.min())
    data = data * 255
    data = data.astype(np.uint8)
    return data, crop

def get_label(arr):
    if 1 in arr:
        red.append(1)
    else:
        red.append(0)
        
    if 2 in arr:
        pink.append(1)
    else:
        pink.append(0)
         
    if 3 in arr:
        yellow.append(1)
    else:
        yellow.append(0)
        
    if 4 in arr:
        blue.append(1)
    else:
        blue.append(0)   

def have_back(image):
    background_value=0
    tolerance=0.00001
    is_foreground = np.logical_or(image < -1001,
                                  image > -400)
    foreground = np.zeros(is_foreground.shape, dtype=np.uint8)
    foreground[is_foreground] = 1
    return foreground

def crop_img(data, rtol=1e-8):

    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)  ##
    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
        
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [(s,e) for s, e in zip(start, end)]
    return slices

def get_crop_index(data):
    foreground = have_back(data)
    crop = crop_img(foreground)
    return crop
    

def save_png(data, outdir, prefix, crop, if_mask = False):
    m = data.shape[0]
    for index in range(m):
        output = os.path.join(outdir, prefix + '_' + str(index) + '.png')
        output_list = output.split('/')
        save_path = '/'.join(output_list[-5:])
        if if_mask:
            mask_path.append(save_path)
            get_label(data[index,:,:])
        else:
            data_path.append(save_path)
            folder.append(int(prefix[-2:]))
            ys.append(crop[1][0])
            ye.append(crop[1][1])
            xs.append(crop[2][0])
            xe.append(crop[2][1])
        cv2.imwrite(output, data[index,:,:])


def processing(tr_list, indir, target_dir, IS_SCALE, IS_SAVE_CSV = True, IS_TEST = True):
    '''
    When generating 2D data, the data information is saved in the form of CSV 
    to facilitate the subsequent network training.
    tr_list: the list of the number of patient. 
    indir: The relative path to which the data is stored.
    target_dir : The relative path to the data store.
    IS_SCALE : Whether to normalize or not.
    IS_SAVE_CSV : Whether to save relevant information in CSV file， if it has already been saved, there is no need to save it.
    IS_TEST : There is no mask in the test dataset, which needs extra processing.
    '''
    global data_path, mask_path, red, pink, yellow, blue, ys, ye, xs, xe, folder
    for x in tr_list:
        sd = str(x).zfill(2)
        image_file = os.path.join(data_foler, indir, 'Patient_' + sd + '.nii.gz')  
        
        print(x)
        mean = mean_csv.loc[mean_csv['folder']== x]['median'].values[0]
        
        image, crop_index = read_image(image_file, mean = mean ,scale = IS_SCALE)
        print(crop_index)
        outdir = os.path.join(data_foler, target_dir, sd)  
        if not os.path.exists(os.path.join(outdir, 'images')):
            os.makedirs(os.path.join(outdir, 'images'))   
        save_png(image, os.path.join(outdir, 'images'), 'Patient_' + sd, crop = crop_index, if_mask = False)
        
        #Mask data is generated if it is not a test dataset 
        if not IS_TEST:
            gt_file = os.path.join(data_foler, indir, 'GT_' + sd + '.nii.gz')
            mask = sitk.ReadImage(gt_file)
            mask = sitk.GetArrayFromImage(mask)
            if not os.path.exists(os.path.join(outdir, 'masks')):
                os.makedirs(os.path.join(outdir, 'masks'))
            save_png(mask, os.path.join(outdir, 'masks'), 'GT_' + sd, crop = crop_index, if_mask = True)
            
    # Save the valid information in a CSV file
    if IS_SAVE_CSV:
        print('Now save the information to csv!')
        if not IS_TEST:
            dic = {'filename': data_path, 'maskname': mask_path,'folder':folder, 'red':red, 'pink':pink, 'yellow':yellow, 'blue':blue,
                  'ys':ys,
                  'ye':ye,
                  'xs':xs,
                  'xe':xe}
            df = pd.DataFrame(dic)
            if 'val' in target_dir:
                df.to_csv('val_info.csv',index = False)
            else:
                df.to_csv('train_info.csv',index = False)
        else:
            dic = {'filename': data_path, 
                   'folder':folder, 
                  'ys':ys,
                  'ye':ye,
                  'xs':xs,
                  'xe':xe}
            df = pd.DataFrame(dic)
            df.to_csv('test_info.csv',index = False)   
    print('Done!')
    print(target_dir)
    
   #reset global value
    data_path = []
    mask_path = []
    red = []
    pink = []
    yellow = []
    blue = []
    ys = []
    ye = []
    xs = []
    xe = []
    folder = [] 
            
    
def main():
    indir = 'segthor_gz/train'
    indir_test = 'segthor_gz/test'
    
    out_test_dir = 'train_flam/test'
    out_train_dir = 'train_flam/train'
    out_validate_dir = 'train_flam/validates'

    IS_SCALE = True #The image HU pixel value is intercepted to [-400, 400] and normalized to [0, 255]

    #for train
    tr_list = [ 5, 33, 26,  2, 13, 27, 20, 40,  6, 19, 36, 18, 35, 22, 11, 28,  7,
            37, 29,  8, 30, 24, 16, 10, 38, 21,  4, 32,  9, 23, 25, 31]
    va_list = [1, 39, 34, 12, 17, 14, 15, 3]

    test_list = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    processing(test_list, 
               indir_test, 
               out_test_dir,     
               IS_SCALE, 
               IS_SAVE_CSV = False, 
               IS_TEST = True) 
    processing(tr_list,   
               indir,      
               out_train_dir,    
               IS_SCALE, 
               IS_SAVE_CSV = False, 
               IS_TEST = False)
    processing(va_list,   
               indir,      
               out_validate_dir, 
               IS_SCALE, 
               IS_SAVE_CSV = False, 
               IS_TEST = False)


main()

