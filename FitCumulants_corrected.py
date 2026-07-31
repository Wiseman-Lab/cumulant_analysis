# hold functions for fitting cumulants to models

import numpy as np
from lmfit import Parameters, minimize, fit_report

def model_cumulants(N, B, gamma, background, two_color=False):
    """
    N: particle density per species
    B: brightness per species
    gamma: shape factor: 1 for first order cumulant, 1/2 for second order cumulants
    cumulants: 
    """
    N = np.array(N)
    B = np.array(B)
    if not two_color:
        B = np.squeeze(B)
        k1 = gamma[0] * np.sum( N * B) + background
        k2 = gamma[1] * np.sum( N * B**2)
        k3 = gamma[2] * np.sum( N * B**3)
        return k1, k2, k3
    else:
        k10 = gamma[0] * np.sum( N * B[:, 0]) + background[0]
        k01 = gamma[1] * np.sum( N * B[:, 1]) + background[1]
        k20 = gamma[2] * np.sum( N * B[:, 0]**2)
        k02 = gamma[3] * np.sum( N * B[:, 1]**2)
        k11 = gamma[4] * np.sum( N * B[:, 0] * B[:, 1] )
        k30 = gamma[5] * np.sum( N * B[:, 0]**3) 
        k03 = gamma[6] * np.sum( N * B[:, 1]**3)
        k21 = gamma[7] * np.sum( N * B[:, 0]**2 * B[:, 1]) 
        k12 = gamma[8] * np.sum( N * B[:, 0] * B[:, 1]**2) 
        return k10, k01, k20, k02, k11, k30, k03, k21, k12
    
    
def check_cumulants_1c(pars, k1, vark1, k2, vark2, k3, vark3, Nspecies):
    vals = pars.valuesdict()
    N = np.zeros(Nspecies)
    B = np.zeros((Nspecies, 1))
    gamma = np.zeros(3)
    for i in range(Nspecies):
        N[i] = vals[f'N{i+1}']
        B[i] = vals[f'B{i+1}']
    for i in range(3):
        gamma[i] = vals[f'gamma{i+1}']
    background = vals['Background_ch1']
    fk1, fk2, fk3 = model_cumulants(N, B, gamma, background, two_color=False)
    res1 = (k1 - fk1)/np.sqrt(vark1)
    res2 = (k2 - fk2)/np.sqrt(vark2)
    res3 = (k3 - fk3)/np.sqrt(vark3)
    res = np.concatenate([res1, res2, res3])
    return res


def check_cumulants_2c(pars, k10, vark10, k01, vark01, k20, vark20, k02, vark02, k11, vark11, k30, vark30, k03, vark03, k21, vark21, k12, vark12, Nspecies):
    vals = pars.valuesdict()
    N = np.zeros(Nspecies)
    B = np.zeros((Nspecies, 2))
    gamma = np.ones(9)
    for i in range(Nspecies):
        N[i] = vals[f'N{i+1}']
        for j, letter in enumerate(['a', 'b']):
            B[i, j] = vals[f'B{i+1}{letter}']
    for i in range(9):
        gamma[i] = vals[f'gamma{i+1}']
    background_ch1 , background_ch2 = vals['Background_ch1'], vals['Background_ch2']
    fk10, fk01, fk20, fk02, fk11, fk30, fk03, fk12, fk21 = model_cumulants(N, B, gamma, [background_ch1, background_ch2], two_color=True)
    res10 = (k10 - fk10)/np.sqrt(vark10)
    res01 = (k01 - fk01)/np.sqrt(vark01)
    res20 = (k20 - fk20)/np.sqrt(vark20)
    res02 = (k02 - fk02)/np.sqrt(vark02)
    res11 = (k11 - fk11)/np.sqrt(vark11)
    res30 = (k30 - fk30)/np.sqrt(vark30)
    res03 = (k03 - fk03)/np.sqrt(vark03)
    res21 = (k21 - fk21)/np.sqrt(vark21)
    res12 = (k12 - fk12)/np.sqrt(vark12)
    res = np.concatenate([res10, res01, res20, res02, res11, res30, res03, res21, res12])
    return res

def gaussian_gamma(e2rad, two_color=True):
    """
    e2rad: e^-2 radius of gaussian psf (in pixel space)
    """
    if two_color:
        gamma = np.ones(9)
        gamma[2] = 1 / np.pi / (e2rad[0]**2)
        gamma[3] = 1 / np.pi / (e2rad[1]**2) 
        gamma[4] = 2 / np.pi / (e2rad[0]**2 + e2rad[1]**2)
        gamma[5] = 4 / 3 / np.pi**2 / e2rad[0]**4
        gamma[6] = 4 / 3 / np.pi**2 / e2rad[1]**4
        gamma[7] = 4 / np.pi**2 / e2rad[0]**2 / (2 * e2rad[1]**2 + e2rad[0]**2)
        gamma[8] = 4 / np.pi**2 / e2rad[1]**2 / (2 * e2rad[0]**2 + e2rad[1]**2)
    else:
        gamma = np.ones(3)
        gamma[1] = 1 / np.pi / (e2rad**2) 
        gamma[2] = 4 / 3 / np.pi**2 / e2rad**4
    return gamma


def fit_cumulants(N0, B0, background, e2rad, Kvals, Kvars, two_color=False, varyN=True, varyB=True, varygamma=False, varyback=False, bmax=5000):
    """
    N0: initial N values for each species
    B0: initial brightness value(s) for each species (in each channel)
    gamma: first and second order shape factor
    background: background noise amplitude (must be > 0 but can be small).
    kvals: np array of cumulants
    Kvars: np array of variances of cumulants
    """
    Nspecies = len(N0)
    pars = Parameters()
    for i in range(Nspecies):
        pars.add(f'N{i+1}', value=N0[i], min=0, max=1000, vary=varyN)
        if two_color:
            for j, letter in enumerate(['a', 'b']):
                pars.add(f'B{i+1}{letter}', value=B0[i, j], min=0, max=bmax, vary=varyB)
        else:
            pars.add(f'B{i+1}', value=B0[i], min=0, max=bmax, vary=varyB)
    gamma = gaussian_gamma(e2rad, two_color)
    for i in range(len(gamma)):
        pars.add(f'gamma{i+1}', value=gamma[i], vary=varygamma)
    pars['gamma1'].value = 1
    pars['gamma1'].vary = False
    if two_color:
        pars['gamma2'].vary = False
    if len(background) == 1:
        background = [background[0], background[0]]
        pars.add('Background_ch1', value=np.float64(background[0]), min=0, max=200, vary=varyback)
    elif len(background) == 2:
        pars.add('Background_ch1', value=np.float64(background[0]), min=0, max=1000, vary=varyback)
        pars.add('Background_ch2', value=np.float64(background[1]), min=0, max=1000, vary=varyback)
    if two_color:
        return minimize(check_cumulants_2c, pars, args=(Kvals[:, 0], Kvars[:, 0], Kvals[:, 1], Kvars[:, 1], Kvals[:, 2], Kvars[:, 2], Kvals[:, 3], Kvars[:, 3], Kvals[:, 4], Kvars[:, 4], Kvals[:, 5], Kvars[:, 5], Kvals[:, 6], Kvars[:, 6], Kvals[:, 7], Kvars[:, 7], Kvals[:, 8], Kvars[:, 8], Nspecies))
    else:
        return minimize(check_cumulants_1c, pars, args=(Kvals[:, 0], Kvars[:, 0], Kvals[:, 1], Kvars[:, 1], Kvals[:, 2], Kvars[:, 2], Nspecies))
    


