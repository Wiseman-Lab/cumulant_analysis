import numpy as np
import tifffile as tif
import pandas as pd
import os

Generator = np.random.default_rng()

def single_color(path, factorial=False, fmap=None, out=False, bootstrap=False):
    """
    Calculate single color cumulants for each image tiff in folder.

    Path (str): path to folder containing images
    factorial (bool): calculate factorial cumulants if single photon counting

    Saves "single_color_cumulants.csv" in image folder
    """
    if isinstance(path, str):
        if os.path.isdir(path):
            files = os.listdir(path)
            images = [f for f in files if (f[-5:] == '.tiff' or f[-4:] == '.lsm' or f[-4:]== '.tif')]
            images = sorted(images)
            cumulants = np.zeros((len(images), 3))
            cum_variances = np.zeros(cumulants.shape)
            for i, imname in enumerate(images):
                print(imname)
                img = tif.imread(path + '\\' + imname) 
                if fmap is not None:
                    img = img[fmap.astype('bool')]
                if bootstrap > 0:
                    img = Generator.choice(img.ravel(), img.size*bootstrap, replace=True)
                n = img.size
                k1 = np.mean(img) # k_1
                # u# are raw cumulants, not to be confused with moments as the u notation might suggest...
                u2 = np.mean( (img - np.mean(img))**2 ) 
                u3 = np.mean( (img - np.mean(img))**3 )
                if factorial:
                    # these are factorial cumulants
                    k2 = np.mean( (img - np.mean(img))**2 ) - np.mean(img)
                    k3 = np.mean((img - np.mean(img))**3) - 3*np.mean((img - np.mean(img))**2) + 2*np.mean(img)
                else:
                    k2 = u2 
                    k3 = u3
                u4 = np.mean( (img - np.mean(img))**4 ) - 3 * (u2**2)
                u5 = np.mean( (img - np.mean(img))**5 ) - 10 * u3 * u2
                u6 = np.mean( (img - np.mean(img))**6 ) - 15*np.mean( (img - np.mean(img))**4 )*u2 - 10*(u3**2) + 30*(u2**3) 
                cumulants[i, 0] = k1
                cumulants[i, 1] = k2
                cumulants[i, 2] = k3
                cum_variances[i, 0] = k2 / n
                if factorial:
                    cum_variances[i, 1] = ( u2 + 2*u2**2 - 3*u3 + u4 ) / n   # these are the correct expressions. (this one from mueller 2004)
                    cum_variances[i, 2] = ( 18*u2**2 + 6*u2**3 - 12*u3 + 9*u3**2 + 13*u4 + u2*(4 - 36*u3 + 9*u4) -6*u5 + u6 ) / n  
                else:
                    cum_variances[i, 1] = (u4 / n) + (2*u2**2)/(n-1)   # this one from wolfram mathworld
                    cum_variances[i, 2] = (u6/n) + (9*u2*u4/(n-1)) + (9*u3**2)/(n-1) + (6*n*u2**3)/(n-1)/(n-2)

        elif os.path.isfile(path):
            images = tif.imread(path)
            if len(images.shape) == 2:
                images = np.array([images])
            elif len(images.shape) > 3:
                print('unexpected image shape')
    
            cumulants = np.zeros((len(images), 3))
            cum_variances = np.zeros(cumulants.shape)

            for i in range(len(images)):
                img = images[i]
                if fmap is not None:
                    img = img[fmap.astype('bool')]
                if bootstrap > 0:
                    img = Generator.choice(img.ravel(), img.size*bootstrap, replace=True)
                n = img.size
                k1 = np.mean(img) # k_1
                # u# are raw cumulants, not to be confused with moments as the u notation might suggest...
                u2 = np.mean( (img - np.mean(img))**2 ) 
                u3 = np.mean( (img - np.mean(img))**3 )
                if factorial:
                    # these are factorial cumulants
                    k2 = np.mean( (img - np.mean(img))**2 ) - np.mean(img)
                    k3 = np.mean((img - np.mean(img))**3) - 3*np.mean((img - np.mean(img))**2) + 2*np.mean(img)
                else:
                    k2 = u2 
                    k3 = u3
                u4 = np.mean( (img - np.mean(img))**4 ) - 3 * (u2**2)
                u5 = np.mean( (img - np.mean(img))**5 ) - 10 * u3 * u2
                u6 = np.mean( (img - np.mean(img))**6 ) - 15*np.mean( (img - np.mean(img))**4 )*u2 - 10*(u3**2) + 30*(u2**3) 
                cumulants[i, 0] = k1
                cumulants[i, 1] = k2
                cumulants[i, 2] = k3
                cum_variances[i, 0] = k2 / n
                if factorial:
                    cum_variances[i, 1] = ( u2 + 2*u2**2 - 3*u3 + u4 ) / n   # these are the correct expressions. (this one from mueller 2004)
                    cum_variances[i, 2] = ( 18*u2**2 + 6*u2**3 - 12*u3 + 9*u3**2 + 13*u4 + u2*(4 - 36*u3 + 9*u4) -6*u5 + u6 ) / n  
                else:
                    cum_variances[i, 1] = (u4 / n) + (2*u2**2)/(n-1)   # this one from wolfram mathworld
                    cum_variances[i, 2] = (u6/n) + (9*u2*u4/(n-1)) + (9*u3**2)/(n-1) + (6*n*u2**3)/(n-1)/(n-2)
        
    elif isinstance(path, np.ndarray):
        images = path.copy()
        if len(path.shape) == 2:
            images = np.array([images])
        cumulants = np.zeros((len(images), 3))
        cum_variances = np.zeros(cumulants.shape)
        for i in range(len(images)):
            img = images[i]
            if fmap is not None:
                img = img[fmap.astype('bool')]
            if bootstrap > 0:
                img = Generator.choice(img.ravel(), img.size*bootstrap, replace=True)
            n = img.size
            k1 = np.mean(img) # k_1
            # u# are raw cumulants, not to be confused with moments as the u notation might suggest...
            u2 = np.mean( (img - np.mean(img))**2 ) 
            u3 = np.mean( (img - np.mean(img))**3 )
            if factorial:
                # these are factorial cumulants
                k2 = np.mean( (img - np.mean(img))**2 ) - np.mean(img)
                k3 = np.mean((img - np.mean(img))**3) - 3*np.mean((img - np.mean(img))**2) + 2*np.mean(img)
            else:
                k2 = u2 
                k3 = u3
            u4 = np.mean( (img - np.mean(img))**4 ) - 3 * (u2**2)
            u5 = np.mean( (img - np.mean(img))**5 ) - 10 * u3 * u2
            u6 = np.mean( (img - np.mean(img))**6 ) - 15*np.mean( (img - np.mean(img))**4 )*u2 - 10*(u3**2) + 30*(u2**3) 
            cumulants[i, 0] = k1
            cumulants[i, 1] = k2
            cumulants[i, 2] = k3
            cum_variances[i, 0] = k2 / n
            if factorial:
                cum_variances[i, 1] = ( u2 + 2*u2**2 - 3*u3 + u4 ) / n   # these are the correct expressions. (this one from mueller 2004)
                cum_variances[i, 2] = ( 18*u2**2 + 6*u2**3 - 12*u3 + 9*u3**2 + 13*u4 + u2*(4 - 36*u3 + 9*u4) -6*u5 + u6 ) / n  
            else:
                cum_variances[i, 1] = (u4 / n) + (2*u2**2)/(n-1)   # this one from wolfram mathworld
                cum_variances[i, 2] = (u6/n) + (9*u2*u4/(n-1)) + (9*u3**2)/(n-1) + (6*n*u2**3)/(n-1)/(n-2)
    
    if out:
        return cumulants, cum_variances
    else:
        df = pd.DataFrame(images, columns=['Images'])
        df['k_1'] = cumulants[:, 0]
        df['k_2'] = cumulants[:, 1]
        df['k_3'] = cumulants[:, 2]
        df['var(k_1)'] = cum_variances[:, 0]
        df['var(k_2)'] = cum_variances[:, 1]
        df['var(k_3)'] = cum_variances[:, 2]
        df.to_csv(path + '\\' + 'single_color_cumulants.csv')

def two_color(path, factorial=False, fmap=None, out=False, bootstrap=False):
    """
    Calculate single color cumulants for each image tiff in folder or for each frame of an image stack.

    stack organization dimensions should always be: [frame, channels,  

    Path (str): path to folder containing images
    factorial (bool): calculate factorial cumulants if single photon counting

    In the equations in this function, generally u_n,m denotes the cumulant and k_n,m is the factorial cumulant. Sometimes these overlap.

    Saves "two_color_cumulants.csv" in image folder
    """
    if isinstance(path, str):
        if os.path.isdir(path):
            files = os.listdir(path)
            images = [f for f in files if  (f[-5:] == '.tiff' or f[-4:] == '.lsm' or f[-4:]== '.tif')]
            images = sorted(images)
            cumulants = np.zeros([len(images), 9])
            cum_variances = np.zeros(cumulants.shape)

            for i, imname in enumerate(images):
                img = tif.imread(path + '\\' + imname)
                # for hyperspectral images, add bins together into 2 channels 
                im1 = img[0]
                im2 = img[1]
                if fmap is not None:
                    im1 = im1[fmap.astype('bool')]
                    im2 = im2[fmap.astype('bool')]
                if bootstrap > 0:
                    im1 = Generator.choice(im1.ravel(), im1.size*bootstrap, replace=True)
                    im2 = Generator.choice(im2.ravel(), im2.size*bootstrap, replace=True)
                n = im1.size
                k10 = np.mean(im1) # k_1,0
                k01 = np.mean(im2) # k_0,1
                # these are raw cumulants! (equivalent to central moments though)
                u20 = np.mean( (im1 - np.mean(im1))**2 )  
                u02 = np.mean( (im2 - np.mean(im2))**2 )  
                u30 = np.mean( (im1 - np.mean(im1))**3 )
                u03 = np.mean( (im2 - np.mean(im2))**3 ) 


                u40 = np.mean( (im1 - np.mean(im1))**4 ) - 3*u20**2
                u04 = np.mean( (im2 - np.mean(im2))**4 ) - 3*u02**2
                u50 = np.mean( (im1 - np.mean(im1))**5 ) - 10 * u30 * u20
                u05 = np.mean( (im2 - np.mean(im2))**5 ) - 10 * u03 * u02
                u60 = np.mean( (im1 - np.mean(im1))**6 ) - 15*np.mean( (im1 - np.mean(im1))**4 )*u20 - 10*(u30**2) + 30*(u20**3) 
                u06 = np.mean( (im2 - np.mean(im2))**6 ) - 15*np.mean( (im2 - np.mean(im2))**4 )*u02 - 10*(u03**2) + 30*(u02**3) 
                
                k11 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2)) ) # k_1,1
                u12 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2))**2 )
                u21 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2)) )

                u22 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**2 ) 
                k22 = u22 + u20*u02 + k11**2
                u31 = np.mean( (im1 - np.mean(im1))**3 * (im2 - np.mean(im2)) )
                u13 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2))**3 )
                u32 = np.mean( (im1 - np.mean(im1))**3 * (im2 - np.mean(im2))**2 )
                u23 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**3 )
                u42 = np.mean( (im1 - np.mean(im1))**4 * (im2 - np.mean(im2))**2 )
                u24 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**4 )
                k31 = u31 - 3*k11*u20
                k13 = u13 - 3*k11*u02
                k32 = u32 - 6*k11*u21 - 3*u20*u12 - u02*u30
                k23 = u23 - 6*k11*u12 - 3*u02*u21 - u20*u03
                k40 = u40 - 3*u20**2
                k04 = u04 - 3*u02**2
                k42 = 6*u02*u20**2 + 24*u20*k11**2 - 4*u12*u30 - 6*u21**2 - 8*u31*k11 - 6*u20*u22 -u40*u02 + u42
                k24 = 6*u20*u02**2 + 24*u02*k11**2 - 4*u21*u03 - 6*u12**2 - 8*u13*k11 - 6*u02*u22 -u04*u20 + u24
                # this is where Im at

                if factorial:
                    cum_variances[i, 0] = u20 / n
                    cum_variances[i, 1] = u02 / n
                    cum_variances[i, 2] = ( u20 + 2*u20**2 - 3*u30 + u40 ) / n 
                    cum_variances[i, 3] = ( u02 + 2*u02**2 - 3*u03 + u04 ) / n 
                    cum_variances[i, 4] = ( k11**2 + u02*u20 + k22 ) / n
                    cum_variances[i, 5] = ( 18*u20**2 + 6*u20**3 - 12*u30 + 9*u30**2 + 13*u40 + u20*(4 - 36*u30 + 9*u40) -6*u50 + u60 ) / n  
                    cum_variances[i, 6] = ( 18*u02**2 + 6*u02**3 - 12*u03 + 9*u03**2 + 13*u04 + u02*(4 - 36*u03 + 9*u04) -6*u05 + u06 ) / n  
                    cum_variances[i, 7] = ( (k11**2 + u02*u20 + k22) + (4*u20*k11**2 + 2*u02*u20**2 + 5*u21**2 + 4*u20*k22 + 4*u12*u30 + 4*k11*k31 + u02*k40 + k42) + 2*(2*u12*u20 + 3*k11*u21 + u02*u30 + k32) ) / n
                    cum_variances[i, 8] = ( (k11**2 + u02*u20 + k22) + (4*u02*k11**2 + 2*u20*u02**2 + 5*u12**2 + 4*u02*k22 + 4*u21*u03 + 4*k11*k13 + u20*k04 + k24) + 2*(2*u21*u02 + 3*k11*u12 + u20*u03 + k23) ) / n
                else:
                    cum_variances[i, 0] = u20 / n
                    cum_variances[i, 1] = u02 / n
                    cum_variances[i, 2] = (u40 / n) + (2 * u20**2 / (n-1))
                    cum_variances[i, 3] = (u04 / n) + (2 * u02**2 / (n-1))
                    cum_variances[i, 4] = (k11**2 + u02*u20 + k22) / n
                    cum_variances[i, 5] = (u60/n) + (9*u20*u40/(n-1)) + (9*u30**2)/(n-1) + (6*n*u20**3)/(n-1)/(n-2)
                    cum_variances[i, 6] = (u06/n) + (9*u02*u04/(n-1)) + (9*u03**2)/(n-1) + (6*n*u02**3)/(n-1)/(n-2)
                    cum_variances[i, 7] = ( 4*u20*k11**2 + 2*u02*u20**2 + 5*u21**2 + 4*u20*k22 + 4*u12*u30 + 4*k11*k31 + u02*k40 + k42) / n
                    cum_variances[i, 8] = ( 4*u02*k11**2 + 2*u20*u02**2 + 5*u12**2 + 4*u02*k22 + 4*u21*u03 + 4*k11*k13 + u20*k04 + k24) / n

                if factorial:
                    k20 = u20 - k10 # k_2,0 
                    k02 = u02 - k01 # k_0,2
                    k30 = u30 - 3*u20 + 2*k10
                    k03 = u03 - 3*u02 + 2*k01
                    k21 = u21 - k11
                    k12 = u12 - k11
                else:
                    k20 = u20  # k_2,0 
                    k02 = u02  # k_0,2
                    k30 = u30
                    k03 = u03 
                    k21 = u21
                    k12 = u12

                cumulants[i, 0] = k10
                cumulants[i, 1] = k01
                cumulants[i, 2] = k20
                cumulants[i, 3] = k02
                cumulants[i, 4] = k11
                cumulants[i, 5] = k30
                cumulants[i, 6] = k03
                cumulants[i, 7] = k21
                cumulants[i, 8] = k12

        elif os.path.isfile(path):
            images = tif.imread(path)
            if len(images.shape) == 3:
                images = np.array([images])
            elif len(images.shape) > 4:
                print('unexpected image shape')
        
            cumulants = np.zeros([len(images), 9])
            cum_variances = np.zeros(cumulants.shape)

            for i in range(len(images)):
                img = images[i]
                # for hyperspectral images, add bins together into 2 channels 
                im1 = img[0]
                im2 = img[1]
                if fmap is not None:
                    im1 = im1[fmap.astype('bool')]
                    im2 = im2[fmap.astype('bool')]
                if bootstrap > 0:
                    im1 = Generator.choice(im1.ravel(), im1.size*bootstrap, replace=True)
                    im2 = Generator.choice(im2.ravel(), im2.size*bootstrap, replace=True)
                n = im1.size
                k10 = np.mean(im1) # k_1,0
                k01 = np.mean(im2) # k_0,1
                # these are raw cumulants! (equivalent to central moments though)
                u20 = np.mean( (im1 - np.mean(im1))**2 )  
                u02 = np.mean( (im2 - np.mean(im2))**2 )  
                u30 = np.mean( (im1 - np.mean(im1))**3 )
                u03 = np.mean( (im2 - np.mean(im2))**3 ) 


                u40 = np.mean( (im1 - np.mean(im1))**4 ) - 3*u20**2
                u04 = np.mean( (im2 - np.mean(im2))**4 ) - 3*u02**2
                u50 = np.mean( (im1 - np.mean(im1))**5 ) - 10 * u30 * u20
                u05 = np.mean( (im2 - np.mean(im2))**5 ) - 10 * u03 * u02
                u60 = np.mean( (im1 - np.mean(im1))**6 ) - 15*np.mean( (im1 - np.mean(im1))**4 )*u20 - 10*(u30**2) + 30*(u20**3) 
                u06 = np.mean( (im2 - np.mean(im2))**6 ) - 15*np.mean( (im2 - np.mean(im2))**4 )*u02 - 10*(u03**2) + 30*(u02**3) 
                
                k11 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2)) ) # k_1,1
                u12 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2))**2 )
                u21 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2)) )

                u22 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**2 ) 
                k22 = u22 + u20*u02 + k11**2
                u31 = np.mean( (im1 - np.mean(im1))**3 * (im2 - np.mean(im2)) )
                u13 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2))**3 )
                u32 = np.mean( (im1 - np.mean(im1))**3 * (im2 - np.mean(im2))**2 )
                u23 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**3 )
                u42 = np.mean( (im1 - np.mean(im1))**4 * (im2 - np.mean(im2))**2 )
                u24 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**4 )
                k31 = u31 - 3*k11*u20
                k13 = u13 - 3*k11*u02
                k32 = u32 - 6*k11*u21 - 3*u20*u12 - u02*u30
                k23 = u23 - 6*k11*u12 - 3*u02*u21 - u20*u03
                k40 = u40 - 3*u20**2
                k04 = u04 - 3*u02**2
                k42 = 6*u02*u20**2 + 24*u20*k11**2 - 4*u12*u30 - 6*u21**2 - 8*u31*k11 - 6*u20*u22 -u40*u02 + u42
                k24 = 6*u20*u02**2 + 24*u02*k11**2 - 4*u21*u03 - 6*u12**2 - 8*u13*k11 - 6*u02*u22 -u04*u20 + u24
                # this is where Im at

                if factorial:
                    cum_variances[i, 0] = u20 / n
                    cum_variances[i, 1] = u02 / n
                    cum_variances[i, 2] = ( u20 + 2*u20**2 - 3*u30 + u40 ) / n 
                    cum_variances[i, 3] = ( u02 + 2*u02**2 - 3*u03 + u04 ) / n 
                    cum_variances[i, 4] = ( k11**2 + u02*u20 + k22 ) / n
                    cum_variances[i, 5] = ( 18*u20**2 + 6*u20**3 - 12*u30 + 9*u30**2 + 13*u40 + u20*(4 - 36*u30 + 9*u40) -6*u50 + u60 ) / n  
                    cum_variances[i, 6] = ( 18*u02**2 + 6*u02**3 - 12*u03 + 9*u03**2 + 13*u04 + u02*(4 - 36*u03 + 9*u04) -6*u05 + u06 ) / n  
                    cum_variances[i, 7] = ( (k11**2 + u02*u20 + k22) + (4*u20*k11**2 + 2*u02*u20**2 + 5*u21**2 + 4*u20*k22 + 4*u12*u30 + 4*k11*k31 + u02*k40 + k42) + 2*(2*u12*u20 + 3*k11*u21 + u02*u30 + k32) ) / n
                    cum_variances[i, 8] = ( (k11**2 + u02*u20 + k22) + (4*u02*k11**2 + 2*u20*u02**2 + 5*u12**2 + 4*u02*k22 + 4*u21*u03 + 4*k11*k13 + u20*k04 + k24) + 2*(2*u21*u02 + 3*k11*u12 + u20*u03 + k23) ) / n
                else:
                    cum_variances[i, 0] = u20 / n
                    cum_variances[i, 1] = u02 / n
                    cum_variances[i, 2] = (u40 / n) + (2 * u20**2 / (n-1))
                    cum_variances[i, 3] = (u04 / n) + (2 * u02**2 / (n-1))
                    cum_variances[i, 4] = (k11**2 + u02*u20 + k22) / n
                    cum_variances[i, 5] = (u60/n) + (9*u20*u40/(n-1)) + (9*u30**2)/(n-1) + (6*n*u20**3)/(n-1)/(n-2)
                    cum_variances[i, 6] = (u06/n) + (9*u02*u04/(n-1)) + (9*u03**2)/(n-1) + (6*n*u02**3)/(n-1)/(n-2)
                    cum_variances[i, 7] = ( 4*u20*k11**2 + 2*u02*u20**2 + 5*u21**2 + 4*u20*k22 + 4*u12*u30 + 4*k11*k31 + u02*k40 + k42) / n
                    cum_variances[i, 8] = ( 4*u02*k11**2 + 2*u20*u02**2 + 5*u12**2 + 4*u02*k22 + 4*u21*u03 + 4*k11*k13 + u20*k04 + k24) / n

                if factorial:
                    k20 = u20 - k10 # k_2,0 
                    k02 = u02 - k01 # k_0,2
                    k30 = u30 - 3*u20 + 2*k10
                    k03 = u03 - 3*u02 + 2*k01
                    k21 = u21 - k11
                    k12 = u12 - k11
                else:
                    k20 = u20  # k_2,0 
                    k02 = u02  # k_0,2
                    k30 = u30
                    k03 = u03 
                    k21 = u21
                    k12 = u12

                cumulants[i, 0] = k10
                cumulants[i, 1] = k01
                cumulants[i, 2] = k20
                cumulants[i, 3] = k02
                cumulants[i, 4] = k11
                cumulants[i, 5] = k30
                cumulants[i, 6] = k03
                cumulants[i, 7] = k21
                cumulants[i, 8] = k12

    elif isinstance(path, np.ndarray):
        images = np.squeeze(path)
        if len(images.shape) == 3: # this case should only be 2 channel with 2d images (i.e. ch * x * y)
            images = np.array([images])
        if len(images.shape) == 2:  # this case should only be 2 channel by ALREADY SEGMENTED 1d vector of pixels (i.e. ch * Npixels in map)
            images = np.array([images])
        cumulants = np.zeros([len(images), 9])
        cum_variances = np.zeros(cumulants.shape)

        for i in range(len(images)):
            img = images[i]
            # for hyperspectral images, add bins together into 2 channels 
            im1 = img[0]
            im2 = img[1]
            if fmap is not None:
                im1 = im1[fmap.astype('bool')]
                im2 = im2[fmap.astype('bool')]
            if bootstrap > 0:
                im1 = Generator.choice(im1.ravel(), im1.size*bootstrap, replace=True)
                im2 = Generator.choice(im2.ravel(), im2.size*bootstrap, replace=True)
            n = im1.size
            k10 = np.mean(im1) # k_1,0
            k01 = np.mean(im2) # k_0,1
            # these are raw cumulants! (equivalent to central moments though)
            u20 = np.mean( (im1 - np.mean(im1))**2 )  
            u02 = np.mean( (im2 - np.mean(im2))**2 )  
            u30 = np.mean( (im1 - np.mean(im1))**3 )
            u03 = np.mean( (im2 - np.mean(im2))**3 ) 


            u40 = np.mean( (im1 - np.mean(im1))**4 ) - 3*u20**2
            u04 = np.mean( (im2 - np.mean(im2))**4 ) - 3*u02**2
            u50 = np.mean( (im1 - np.mean(im1))**5 ) - 10 * u30 * u20
            u05 = np.mean( (im2 - np.mean(im2))**5 ) - 10 * u03 * u02
            u60 = np.mean( (im1 - np.mean(im1))**6 ) - 15*np.mean( (im1 - np.mean(im1))**4 )*u20 - 10*(u30**2) + 30*(u20**3) 
            u06 = np.mean( (im2 - np.mean(im2))**6 ) - 15*np.mean( (im2 - np.mean(im2))**4 )*u02 - 10*(u03**2) + 30*(u02**3) 
            
            k11 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2)) ) # k_1,1
            u12 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2))**2 )
            u21 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2)) )

            u22 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**2 ) 
            k22 = u22 + u20*u02 + k11**2
            u31 = np.mean( (im1 - np.mean(im1))**3 * (im2 - np.mean(im2)) )
            u13 = np.mean( (im1 - np.mean(im1)) * (im2 - np.mean(im2))**3 )
            u32 = np.mean( (im1 - np.mean(im1))**3 * (im2 - np.mean(im2))**2 )
            u23 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**3 )
            u42 = np.mean( (im1 - np.mean(im1))**4 * (im2 - np.mean(im2))**2 )
            u24 = np.mean( (im1 - np.mean(im1))**2 * (im2 - np.mean(im2))**4 )
            k31 = u31 - 3*k11*u20
            k13 = u13 - 3*k11*u02
            k32 = u32 - 6*k11*u21 - 3*u20*u12 - u02*u30
            k23 = u23 - 6*k11*u12 - 3*u02*u21 - u20*u03
            k40 = u40 - 3*u20**2
            k04 = u04 - 3*u02**2
            k42 = 6*u02*u20**2 + 24*u20*k11**2 - 4*u12*u30 - 6*u21**2 - 8*u31*k11 - 6*u20*u22 -u40*u02 + u42
            k24 = 6*u20*u02**2 + 24*u02*k11**2 - 4*u21*u03 - 6*u12**2 - 8*u13*k11 - 6*u02*u22 -u04*u20 + u24
            # this is where Im at

            if factorial:
                cum_variances[i, 0] = u20 / n
                cum_variances[i, 1] = u02 / n
                cum_variances[i, 2] = ( u20 + 2*u20**2 - 3*u30 + u40 ) / n 
                cum_variances[i, 3] = ( u02 + 2*u02**2 - 3*u03 + u04 ) / n 
                cum_variances[i, 4] = ( k11**2 + u02*u20 + k22 ) / n
                cum_variances[i, 5] = ( 18*u20**2 + 6*u20**3 - 12*u30 + 9*u30**2 + 13*u40 + u20*(4 - 36*u30 + 9*u40) -6*u50 + u60 ) / n  
                cum_variances[i, 6] = ( 18*u02**2 + 6*u02**3 - 12*u03 + 9*u03**2 + 13*u04 + u02*(4 - 36*u03 + 9*u04) -6*u05 + u06 ) / n  
                cum_variances[i, 7] = ( (k11**2 + u02*u20 + k22) + (4*u20*k11**2 + 2*u02*u20**2 + 5*u21**2 + 4*u20*k22 + 4*u12*u30 + 4*k11*k31 + u02*k40 + k42) + 2*(2*u12*u20 + 3*k11*u21 + u02*u30 + k32) ) / n
                cum_variances[i, 8] = ( (k11**2 + u02*u20 + k22) + (4*u02*k11**2 + 2*u20*u02**2 + 5*u12**2 + 4*u02*k22 + 4*u21*u03 + 4*k11*k13 + u20*k04 + k24) + 2*(2*u21*u02 + 3*k11*u12 + u20*u03 + k23) ) / n
            else:
                cum_variances[i, 0] = u20 / n
                cum_variances[i, 1] = u02 / n
                cum_variances[i, 2] = (u40 / n) + (2 * u20**2 / (n-1))
                cum_variances[i, 3] = (u04 / n) + (2 * u02**2 / (n-1))
                cum_variances[i, 4] = (k11**2 + u02*u20 + k22) / n
                cum_variances[i, 5] = (u60/n) + (9*u20*u40/(n-1)) + (9*u30**2)/(n-1) + (6*n*u20**3)/(n-1)/(n-2)
                cum_variances[i, 6] = (u06/n) + (9*u02*u04/(n-1)) + (9*u03**2)/(n-1) + (6*n*u02**3)/(n-1)/(n-2)
                cum_variances[i, 7] = ( 4*u20*k11**2 + 2*u02*u20**2 + 5*u21**2 + 4*u20*k22 + 4*u12*u30 + 4*k11*k31 + u02*k40 + k42) / n
                cum_variances[i, 8] = ( 4*u02*k11**2 + 2*u20*u02**2 + 5*u12**2 + 4*u02*k22 + 4*u21*u03 + 4*k11*k13 + u20*k04 + k24) / n

            if factorial:
                k20 = u20 - k10 # k_2,0 
                k02 = u02 - k01 # k_0,2
                k30 = u30 - 3*u20 + 2*k10
                k03 = u03 - 3*u02 + 2*k01
                k21 = u21 - k11
                k12 = u12 - k11
            else:
                k20 = u20  # k_2,0 
                k02 = u02  # k_0,2
                k30 = u30
                k03 = u03 
                k21 = u21
                k12 = u12

            cumulants[i, 0] = k10
            cumulants[i, 1] = k01
            cumulants[i, 2] = k20
            cumulants[i, 3] = k02
            cumulants[i, 4] = k11
            cumulants[i, 5] = k30
            cumulants[i, 6] = k03
            cumulants[i, 7] = k21
            cumulants[i, 8] = k12

    else:
        print('Un-recognized input type.\n "path" arg can be path to dir, path to tiff file, or numpy ndarray')
        return None
    
    if out:
        return cumulants, cum_variances
    else:
        df = pd.DataFrame(images, columns=['Images'])
        df['k_1,0'] = cumulants[:, 0]
        df['k_0,1'] = cumulants[:, 1]
        df['k_2,0'] = cumulants[:, 2]
        df['k_0,2'] = cumulants[:, 3]
        df['k_1,1'] = cumulants[:, 4]
        df['k_3,0'] = cumulants[:, 5]
        df['k_0,3'] = cumulants[:, 6]
        df['k_2,1'] = cumulants[:, 7]
        df['k_1,2'] = cumulants[:, 8]
        df['var(k_1,0)'] = cum_variances[:, 0]
        df['var(k_0,1)'] = cum_variances[:, 1]
        df['var(k_2,0)'] = cum_variances[:, 2] 
        df['var(k_0,2)'] = cum_variances[:, 3]   
        df['var(k_1,1)'] = cum_variances[:, 4]
        df['var(k_3,0)'] = cum_variances[:, 5] 
        df['var(k_0,3)'] = cum_variances[:, 6]
        df['var(k_2,1)'] = cum_variances[:, 7]
        df['var(k_1,2)'] = cum_variances[:, 8]
        df.to_csv(path + '\\' + 'two_color_cumulants.csv')