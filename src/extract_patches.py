
import glob
import os

import scipy.io as sio

import cv2
import numpy as np
import itertools

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out 
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def draw_contours(mask, ann_inst, line_thickness=2):
    overlay = np.copy((mask).astype(np.uint8))

    label_map = ann_inst
    instances_list = list(np.unique(label_map))  # get list of instances
    instances_list.remove(0)  # remove background
    contours = []
    for inst_id in instances_list:
        instance_map = np.array(
            ann_inst == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = bounding_box(instance_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= ann_inst.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= ann_inst.shape[0] - 1 else y2
        inst_map_crop = instance_map[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        index_correction = np.asarray([[[[x1, y1]]]])
        for i in range(len(contours_crop[0])):
            contours.append(
                list(np.asarray(contours_crop[0][i].astype('int32')) + index_correction))
    contours = list(itertools.chain(*contours))
    cv2.drawContours(overlay, np.asarray(contours), -1, 2, line_thickness)
    return overlay

###########################################################################
if __name__ == '__main__':
    
    cfg = Config()

    extract_type = 'mirror' # 'valid' for fcn8 segnet etc.
                            # 'mirror' for u-net etc.
    # check the patch_extractor.py 'main' to see the different

    # orignal size (win size) - input size - output size (step size)
    # 512x512 - 256x256 - 256x256 fcn8, dcan, segnet
    # 536x536 - 268x268 - 84x84   unet, dist
    # 540x540 - 270x270 - 80x80   xy, hover
    # 504x504 - 252x252 - 252x252 micronet
    step_size = [160, 160] # should match self.train_mask_shape (config.py) 
    win_size  = [400, 400] # should be at least twice time larger than 
                           # self.train_base_shape (config.py) to reduce 
                           # the padding effect during augmentation

    xtractor = PatchExtractor(win_size, step_size)

    ### Paths to data - these need to be modified according to where the original data is stored
    img_ext = '.png'
    img_dir = '/home1/gzy/NucleiSegmentation/CoNSeP/Train/Images/'
    ann_dir = '/home1/gzy/NucleiSegmentation/CoNSeP/Train/Labels/'
    map_dir = '/home1/gzy/NucleiSegmentation/CoNSeP/Train/Maps/'
    #pre_dir = '/home1/gzy/NucleiSegmentation/CoNSeP/Train/Predicts/'
    ####
    out_dir = "/home1/gzy/NucleiSegmentation/CoNSeP/Train/%dx%d_%dx%d_pred_mask" % \
                        (win_size[0], win_size[1], step_size[0], step_size[1])

    file_list = glob.glob('%s/*%s' % (img_dir, img_ext))
    file_list.sort() 

    rm_n_mkdir(out_dir)
    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        print(filename)

        img = cv2.imread(img_dir + basename + img_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        # load dual guassian maps
        positive_map = cv2.imread(map_dir + "positive/" + basename + img_ext, cv2.IMREAD_GRAYSCALE)
        negative_map = cv2.imread(map_dir + "negative/" + basename + img_ext, cv2.IMREAD_GRAYSCALE)
        guassian_map = np.dstack([positive_map, negative_map])
        
        if cfg.type_classification:
            # assumes that ann is HxWx2 (nuclei class labels are available at index 1 of C) 
            ann = sio.loadmat(ann_dir + basename + '.mat')
            ann_inst = ann['inst_map']
            ann_type = ann['type_map']
            
            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            assert np.max(ann_type) <= cfg.nr_types-1, \
                            "Only %d types of nuclei are defined for training"\
                            "but there are %d types found in the input image." % (cfg.nr_types, np.max(ann_type)) 

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype('int32')             
        else:
            # assumes that ann is HxW
            ann_inst = sio.loadmat(ann_dir + basename + '.mat')
            ann_inst = (ann_inst['inst_map']).astype('int32')
            ann = np.expand_dims(ann_inst, -1)
        
        if cfg.type_boundray:
            mask = ann_inst.copy()
            mask[mask!=0] = 1
            mask = draw_contours(mask, ann_inst)
            mask = np.expand_dims(mask, -1)
            ann = np.concatenate([ann, mask], axis=-1)
       
        #img = np.concatenate([img, ann], axis=-1)
#         mask = ann_inst.copy()
#         mask[mask!=0] = 255
#         img[...,0] = mask
#         img[...,1] = mask
#         img[...,2] = mask
        if cfg.combine_mask_train:
#             pred = sio.loadmat(pre_dir + basename + '.mat')
#             pred = pred['result']
#             pred_bnd = pred[0][...,:3]
#             pred_mask = pred[0][...,3:4]
            pred_mask = ann_inst.copy()
            pred_mask[pred_mask!=0] = 1
            pred_mask = np.expand_dims(pred_mask, -1)
            img = np.concatenate([img, pred_mask, ann, guassian_map], axis=-1)
        else:
            img = np.concatenate([img, ann, guassian_map], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
