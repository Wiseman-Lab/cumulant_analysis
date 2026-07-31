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


# get image directory
#   check image directory exists
# get save_location dir
#   check save_location dir exists
# recursively search directory for json files
# at each json file
#   load segmentation parameters from json
#   run mcmc cumulant fit on image stack using loaded params
#   store mcmc results in save_location

def get_head_dir():
    dir = input("Image directory: ")
    if os.path.isdir(dir):
        return dir
    else:
        print(f'cannot find directory at {dir}')
        return False

savepath =  r'C:\Users\teole\OneDrive - McGill University\Research stuff\my_images\abberior_new_FPs\Results_second_round\Results_4'


if __name__ == '__main__':
    head_dir = get_head_dir()
    cat = input('number of frames to concatenate (default 1):')

    if cat == '':
        cat = False
    else:
        cat = True

    logger = logging.getLogger(__name__)

    cumulant_files = glob.glob(head_dir + '/**.npz', recursive=True)

    print('\nFound files: ')
    print(cumulant_files)
    print()
    for i, f in enumerate(cumulant_files):
        fdict = np.load(f)
        if cat:
            cumulants = fdict['cat_cumulants']
            variances = fdict['cat_variances']
            Npixels = fdict['cat_Npixels']
        else:
            cumulants = fdict['cumulants']
            variances = fdict['variances']
            Npixels = fdict['Npixels']

        Rvals = np.zeros(len(cumulants))
        Rerrs = np.zeros(len(cumulants))
        Gvals = np.zeros(len(cumulants))
        Gerrs = np.zeros(len(cumulants))
        RGvals = np.zeros(len(cumulants))
        RGerrs = np.zeros(len(cumulants))
        yfpvals = np.zeros(len(cumulants))
        mchvals = np.zeros(len(cumulants))

        # abberior sted brightnesses
        yfpB = 4.44
        mchB = 2.842
        # FP experiment brightnesses
        # venusB = 0.7
        # mchB = 0.4
        # yfpB = venusB

        e2rad = [3.02213117, 2.32541009]

        try:
            for i in range(len(cumulants)):
                # LSfit = fc.fit_cumulants(np.array([1, 1, 1]), np.array([[yfpB, 0], [0, mchB], [yfpB, mchB]]), np.array([1, 0.05397403, 0.00400625]), [0], np.atleast_2d(cumulants[i]), np.atleast_2d(variances[i]), two_color=True, varyN=True, varyB=False, varyback=False, varygamma=False)
                LSfit = fc.fit_cumulants(np.array([1.1, 1.1, 1.1]), np.array([[yfpB, 0], [0, mchB], [yfpB, mchB]]), [0, 0], e2rad, np.atleast_2d(cumulants[i]), np.atleast_2d(variances[i]), two_color=True, varyN=True, varyB=False, varyback=False, varygamma=False)
                
                a = LSfit.params['N1'].value
                b = LSfit.params['N2'].value
                c = LSfit.params['N3'].value
                aa = LSfit.params['N1'].stderr
                bb = LSfit.params['N2'].stderr
                cc = LSfit.params['N3'].stderr

                Rvals[i] = a
                Rerrs[i] = aa
                Gvals[i] = b
                Gerrs[i] = bb
                RGvals[i] = c
                RGerrs[i] = cc    
                yfpvals[i] = yfpB
                mchvals[i] = mchB
            # patterndate = r'202\d+'
            # patternim = r'binned_tiffs\\.+.json'

            # date = re.search(patterndate, js).group()
            # name = re.search(patternim, js).group()[13:-5]
            # np.savez(savepath + '\\' + date + name + r'_cat1_LS', Rvals=Rvals, Gvals=Gvals, RGvals=RGvals, yfpvals=yfpvals, mchvals=mchvals, Npixels=Npixels)
            idx = re.search(r'\\', f[::-1]).span()[0]
            np.savez(savepath + '\\' + f[-idx:-5] + r'_LS_fit', Rvals=Rvals, Rerrs=Rerrs, Gvals=Gvals, Gerrs=Gerrs, RGvals=RGvals, RGerrs=RGerrs, yfpvals=yfpvals, mchvals=mchvals, Npixels=Npixels)

        except Exception as e:
            print(e)
            print(f'file {i} bad fit:', f)



        








