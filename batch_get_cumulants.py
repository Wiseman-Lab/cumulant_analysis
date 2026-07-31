import numpy as np
import calculateCumulants as cc
import matplotlib.pyplot as plt
import tifffile as tif
import FitCumulants_corrected as fc
import os 
import glob
import skimage.filters as filt
import skimage.morphology as morph
import bayes_fit as bf
import json
import logging
import datetime
import re
from sklearn.decomposition import PCA 


# get image directory
#   check image directory exists
# get save_location dir
#   check save_location dir exists
# recursively search directory for json files
# at each json file
#   load segmentation parameters from json
#   run mcmc cumulant fit on image stack using loaded params
#   store mcmc results in save_location

mydirs = [
    r'C:\Users\teo\OneDrive - McGill University\Research stuff\my_images\abberior_new_FPs\20250122\AT1R-G13_binned_tiffs',
    r'C:\Users\teo\OneDrive - McGill University\Research stuff\my_images\abberior_new_FPs\20250122\G13-mScarlet_binned_tiffs',
    r'C:\Users\teo\OneDrive - McGill University\Research stuff\my_images\abberior_new_FPs\20250122\Lyn-Gq_binned_tiffs' 
    ]

def get_head_dir():
    dir = input("Image directory: ")
    if os.path.isdir(dir):
        return dir
    else:
        print(f'cannot find directory at {dir}')
        return False

savepath =  r"C:\Users\teo\OneDrive - McGill University\Research stuff\my_images\abberior_new_FPs\cumulants_set10"


if __name__ == '__main__':
    head_dir = get_head_dir()
    print('calculating cumulants with 1 and 2 frame concatenation ...')
    cat=2

    logger = logging.getLogger(__name__)

    json_files = glob.glob(head_dir + '/**/*.json', recursive=True)

    logging.basicConfig(filename=head_dir+'progress.log', encoding='utf-8', level=logging.INFO)
    logger.info(f'newest analysis run {str(datetime.datetime.today())}')
    logger.info(f'Working in dir {head_dir}')
    print('\nFound files: ')
    logger.info(f'/nfound files {json_files}')
    print(json_files)
    print()
    patterndate = r'202\d+'
    patternim = r'binned_tiffs\\.+.json'
    for js in json_files:
        print('running analysis for:', js)
        logger.info(f'running analysis for file {js}')
        # set params
        with open(js, "r") as jf:
            param_dict = json.load(jf)
        bminy = param_dict['yfp_min_blur']
        bmaxy = param_dict['yfp_max_blur']
        bminr = param_dict['mcherry_min_blur']
        bmaxr = param_dict['mcherry_max_blur']
        image_path = param_dict['image_stack']
        onlyred=False
        onlyyellow=False
        if 'use_red_map' in param_dict.keys():
            onlyred=True
        if 'use_yellow_map' in param_dict.keys():
            onlyyellow=True
        image_set = tif.imread(image_path)

        cumulants = np.zeros((image_set.shape[1], 9))
        variances = np.zeros(cumulants.shape)
        Npixels = np.zeros(len(cumulants))
        
        cat_cumulants = np.zeros((image_set.shape[1]//cat, 9))
        cat_variances = np.zeros(cat_cumulants.shape)
        cat_Npixels = np.zeros(len(cat_cumulants))
        # segment images and calculate cumulants
        for i in range(image_set.shape[1]):
            yframe = image_set[0, i]
            if onlyred:
                ymap = np.ones(yframe.shape)
            else:
                im0 = yframe
                disk = morph.disk(10)
                prepim = (im0 / np.max(im0) * 35).astype(np.uint8)
                res = filt.rank.entropy(prepim, disk)
                t = filt.threshold_otsu(res)
                fg = np.ones(im0.shape)
                fg[res < t] = 0
                fg = morph.binary_closing(fg, disk)
                fg = morph.binary_opening(fg, disk)
                disk = morph.disk(20)
                fg = morph.binary_erosion(fg, disk)
                segment = im0.copy()
                segment[fg==0] = 0
                blur2 = filt.gaussian(im0, sigma=2, truncate=6)
                blur2[fg==0] = 0 ### this is where I apply the segmentation map
                mapm = np.ones(im0.shape)
                mapm[blur2 <= bminy] = 0
                mapm[ blur2 > bmaxy] = 0
                ymap = mapm.copy()
            # # # # # # # # # # # # # # # # # # # # 
            rframe = image_set[1, i]
            if onlyyellow:
                rmap = np.ones(rframe.shape)
            else:
                im0 = rframe
                disk = morph.disk(10)
                prepim = (im0 / np.max(im0) * 35).astype(np.uint8)
                res = filt.rank.entropy(prepim, disk)
                t = filt.threshold_otsu(res)
                fg = np.ones(im0.shape)
                fg[res < t] = 0
                fg = morph.binary_closing(fg, disk)
                fg = morph.binary_opening(fg, disk)
                disk = morph.disk(20)
                fg = morph.binary_erosion(fg, disk)
                segment = im0.copy()
                segment[fg==0] = 0
                blur2 = filt.gaussian(im0, sigma=2, truncate=6)
                blur2[fg==0] = 0 ### this is where I apply the segmentation map
                mapm = np.ones(im0.shape)
                mapm[blur2 <= bminr] = 0
                mapm[ blur2 > bmaxr] = 0
                rmap = mapm.copy()

            fmap = np.logical_and(ymap, rmap)
            im = image_set[:, i]
   
  
            if i % cat == 0:
                if i > 0:
                    bigim = np.reshape(bigim, (1, bigim.shape[0], bigim.shape[1]))
                    cum, var = cc.two_color(bigim, factorial=True, out=True)
                    cat_cumulants[i//cat] = cum
                    cat_variances[i//cat] = var
                    cat_Npixels[i//cat] = bigim.shape[-1]
                bigim = im[:, fmap.astype('bool')]
            else:
                bigim = np.concatenate([bigim, im[:, fmap.astype('bool')]], axis=1)

            im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
            cum, var = cc.two_color(im, factorial=True, fmap=fmap, out=True)
            cumulants[i] = cum
            variances[i] = var
            Npixels[i] = np.sum(fmap)
        # ###these are for working with lsm images 
        # date = re.search(patterndate, js).group()
        # name = re.search(patternim, js).group()[13:-5]
        # np.savez(savepath + '\\' + date + name + r'_cumulants', cumulants=cumulants, variances=variances, Npixels=Npixels, cat_cumulants=cat_cumulants, cat_variances=cat_variances, cat_Npixels=cat_Npixels)
        
        ###format for abberior images
        idx = re.search(r'\\', image_path[::-1]).span()[0]
        print(idx)
        np.savez(savepath + '\\' + image_path[-idx:-5] + r'_cumulants', cumulants=cumulants, variances=variances, Npixels=Npixels, cat_cumulants=cat_cumulants, cat_variances=cat_variances, cat_Npixels=cat_Npixels)

   