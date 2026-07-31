"""
Bayesian based Fokker-Planck model fitting
"""

import numpy as np
from scipy.integrate import quad
# import numba as nb
# import emcee as mc

  
# K_D dependent fit, fully  simplified version
def log_steady_contKD(x, KD,  NG, NR):
    """
    non-normalized FP equation log scale

    """
    e=1/NG
    q = NR/NG
    B = 2*((NG**2)*q*((e**2)-(e**2)*x) + (NG**2)*((e**2)*(x**2) - (e**2)*x) + KD*NG*(e**2)*x)
    
    f1 = KD * np.log( KD*x + NG*(x-1)*(x-q) ) / NG
    f2 = 2*KD*(KD-(q+1)*NG)*np.arctan((KD-NG*(q-2*x+1)) / (np.sqrt(2*(q+1)*KD*NG-(KD**2)-(NG**2)*((q-1)**2))))
    f3 = NG*np.sqrt(2*(q+1)*KD*NG-(KD**2)-(NG**2)*((q-1)**2))
    return (-NG*(f1 - (f2/f3) - x)) - B

def steady_contKD(x, KD, NG, NR):
    """
    non-normalized FP equation

    """
    e=1/NG
    q = NR/NG
    B = 2*((NG**2)*q*((e**2)-(e**2)*x) + (NG**2)*((e**2)*(x**2) - (e**2)*x) + KD*NG*(e**2)*x)
    
    f1 = KD * np.log( KD*x + NG*(x-1)*(x-q) ) / NG
    f2 = 2*KD*(KD-(q+1)*NG)*np.arctan((KD-NG*(q-2*x+1)) / (np.sqrt(2*(q+1)*KD*NG-(KD**2)-(NG**2)*((q-1)**2))))
    f3 = NG*np.sqrt(2*(q+1)*KD*NG-(KD**2)-(NG**2)*((q-1)**2))
    return np.exp(-NG*(f1 - (f2/f3) - x)) / B

def quad_ZKD(KD, NG, NR):
    """
    normalization for the FP equation
    """
    int_fun = lambda dx : steady_contKD(dx, KD, NG, NR)
    Z, _ = quad(int_fun, 0, 1)
    return Z

def quad_logKD(KD, NG, NR):
    """
    normalization function for log FP equation 
    """
    int_fun = lambda dx : log_steady_contKD(dx, KD, NG, NR)
    norm, _ = quad(int_fun, 0, 1)
    return norm

def cont_distKD(x, KD, NG, NR):
    """
    normalized FP equation
    This is a probabilistic model already!
    """
    Z = quad_ZKD(KD, NG, NR)
    return steady_contKD(x, KD, NG, NR)/Z


def log_likelihood_KD(xdata, KD, NG, NR, maxval=0, logL=True):
    """
    likelihood function for Fokker-Planck model

    **NG should be smaller than NR**

    KD should generally be an array over which the likelihood
    will be evaluated.
    """
    KD_grid = np.atleast_1d(KD) 
    xgrid = np.linspace(0, 1, 200)  # sample grid of xvalues for normalization 
    dK = 1
    if len(KD_grid) > 1:
        dK = KD_grid[1] - KD_grid[0]
    L = np.zeros(len(KD_grid))
    for i, kd in enumerate(KD_grid):
        if logL:
            logfp = log_steady_contKD(xgrid, kd, NG, NR)
            maxv = np.max(logfp)
            diff = maxv - maxval
            pvals = log_steady_contKD(xdata, kd, NG, NR) - diff
            L[i] = np.sum(pvals)
        else:
            pvals = cont_distKD(xdata, kd, NG, NR)
            L[i] = np.product(pvals)
    if not logL:
        L = L / (np.sum(L)*dK)
    return L, KD_grid

def gaussian_prior(mu, sigma, grid):
    pri = np.exp((-(grid - mu)**2)/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
    return pri, grid
    
def uniform_prior(minVal, maxVal, Ngrid=1001):
    grid = np.linspace(minVal, maxVal, Ngrid)
    pri = np.ones(grid.shape) / (maxVal-minVal)
    return grid, pri

### 
# other things

def B(x, kon, koff, NG, NR):
    e=1/NG
    q = NR/NG
    return 1/( 2*(kon*(NG**2)*q*((e**2)-(e**2)*x) + kon*(NG**2)*((e**2)*(x**2) - (e**2)*x) + koff*NG*(e**2)*x))
   
def exponent(x, kon, koff, NG, NR):
    q = NR/NG
    f1 = koff*np.log(NG*kon*(x-1)*(x-q) + koff*x)/(NG*kon)
    f2 = 2*koff*(koff - NG*kon*(q+1))*np.arctan((koff - NG*kon*(q - 2*x + 1))/(np.sqrt(2*NG*koff*kon*(q+1) - ((q-1)**2)*(NG**2)*(kon**2) - (koff**2))))
    f3 = kon*NG*(np.sqrt(2*NG*koff*kon*(q+1) - ((q-1)**2)*(NG**2)*(kon**2) - (koff**2)))
    return -NG*(f1 - (f2/f3) - x)

def get_seed(NG, NR, offset=0, maxval=False):
    """
    get min (and max) K_D values before zeros of the quadratic equation inside the sqrt in the fp model...
    offset val to right/left of leftmost/rightmost root
    maxval = True also returns K_D for rightmost root
    """
    q = NR/NG
    a = -1
    b = 2*(q+1)*NG
    c = -(q-1)**2 * NG**2
    if maxval:
        return (-b + np.sqrt(b**2 - 4*a*c))/(2*a) + offset, (-b - np.sqrt(b**2 - 4*a*c))/(2*a) - offset
    
    return (-b + np.sqrt(b**2 - 4*a*c))/(2*a) + offset # 

"""
often the MAP KD value is too close to an extremity of the sampling grid for KD so confidence intervals can't really be done.
It's definitely not always possible to get a full 95% confidence interval when the KD is almost out of bounds for what can be measured in an image set
BUT, I can still do better than nothing!
"""
"""
    minseed, maxseed = get_seed(N_G, N_R, offset=0, maxval=True)
    KD_grid = np.linspace(minseed, maxseed, int(maxseed-minseed)*4)
    pdf = get_pdf(x_fits, N_G, N_R, KD_grid)

    newbounds:(new min, new max) = get_bounds(pdf, KD_grid)
    newgrid = np.linspace(newbounds[0], newbounds[1], 5000) #5000 here is arbitrary, probably much finer than we need.
    pdf = get_pdf, N_G, N_R, newgrid)

    Plan for what to do:
    - get grid and pdf on grid
    - use coarse grid as a prior to define reasonable edges for fine grid
"""

def get_confidence_interval(pdf, grid, interval):
    cdf = np.cumsum( pdf )*np.diff(grid)[0]
    # return cdf
    kdend = np.argwhere(cdf > (1 - (interval)))[0][0]
    diff = np.inf # wrong offest to start
    diffvals = [0, -1]
    for i, kd in enumerate(grid[:kdend]):
        end = grid[ np.argwhere(cdf > (interval + cdf[i]))[0][0] ]
        if diff > end-kd:
            diff = end-kd
            diffvals = [kd, end]
    return diffvals



####### 

## Bayesian cumulant fitting 
# the commented stuff below here were functions I wrote just to test things. Hopefully the class formalism futher below will be working.
# class formalism works well in MCMC form! it's easier to njit things outside of the class, so all the actual calculations are not in the mcmc runner class.

def get_covar_matrix(cumulants):
    """
    covariance matrix C for multidimensional gaussian fit
    this just calculates the covariance, from a dataset, so each data point still needs to fill in 
    K1, K2, K3 should be arrays of cumulants for similar images (e.g. same time series)
    """
    C = np.cov(np.transpose(cumulants))
    return C

# @nb.njit(parallel=True)
# def p_cumulant_1c(Kvals, Kvars, Cfill, gamma, N, B, back, logp=False):
#     """
#     kvals is a single data vector of 3 cumulants in form [k1, k2, k3]
#     Cfill is the covariance matrix for the single channel cumulants
#     the other things are the normal parameters for the cumulant fit.
#     """
#     C = Cfill.copy()
#     # np.fill_diagonal(C, np.sqrt(Kvars))
#     np.fill_diagonal(C, Kvars)
#     print(N*B)
#     k1 = gamma[0]*np.sum(N*B) + back
#     k2 = gamma[1]*np.sum(N*B**2)
#     k3 = gamma[2]*np.sum(N*B**3)
#     model = np.array([k1, k2, k3])
#     print(np.linalg.eigvals(C))
#     print(np.linalg.det(C))
#     print(np.linalg.inv(C))
#     print(model)
#     if logp:
#         return -0.5*( np.transpose(Kvals - model) @ np.linalg.inv(C) @ (Kvals - model) ) - np.log(np.sqrt(np.abs(np.linalg.det(C)))*(2*np.pi)**(3/2))
#     else:
#         return np.exp(-0.5*( np.transpose(Kvals - model) @ np.linalg.inv(C) @ (Kvals - model) )) / (np.sqrt(np.abs(np.linalg.det(C)))*(2*np.pi)**(3/2))

# def get_covar_matrix_2c(cumulants_2c):
#     c = np.cov(np.transpose(cumulants_2c))
#     return np.real(c)

# @nb.njit(parallel=True)
# def p_cumulant_2c(Kvals, Kvars, Cfill, gamma, N, B, back, logp=False):
#     C = Cfill.copy()
#     np.fill_diagonal(C, Kvars)
#     k10 = gamma[0] * np.sum( N * B[:, 0]) + back[0]
#     k01 = gamma[0] * np.sum( N * B[:, 1]) + back[1]
#     k20 = gamma[1] * np.sum( N * B[:, 0]**2)
#     k02 = gamma[1] * np.sum( N * B[:, 1]**2)
#     k11 = gamma[1] * np.sum( N * B[:, 0] * B[:, 1] )
#     k30 = gamma[2] * np.sum( N * B[:, 0]**3)
#     k03 = gamma[2] * np.sum( N * B[:, 1]**3)
#     k21 = gamma[2] * np.sum( N * B[:, 0]**2 * B[:, 1])
#     k12 = gamma[2] * np.sum( N * B[:, 0] * B[:, 1]**2)
#     model = np.array([k10, k01, k20, k02, k11, k30, k03, k21, k12])
#     print(np.linalg.eigvals(C))
#     if logp:
#         return -0.5*( np.transpose(Kvals - model) @ np.linalg.inv(C) @ (Kvals - model) ) - np.log(np.sqrt(np.abs(np.linalg.det(C)))*(2*np.pi)**(9/2))
#     else:
#         return np.exp(-0.5*( np.transpose(Kvals - model) @ np.linalg.inv(C) @ (Kvals - model) )) / (np.sqrt(np.abs(np.linalg.det(C)))*(2*np.pi)**(9/2))

# @nb.njit()
# def flat_prior(theta):
#     N1 = theta[0]
#     N2 = theta[1]
#     if (N1 < 0) or (N1 > 100):
#         return -np.inf
#     if (N2 < 0) or (N2 > 100):
#         return -np.inf
#     # if (B1 < 1) or (B1 > 20):
#     #     return -np.inf
#     # if (B2 < 1) or (B2 > 20):
#     #     return -np.inf
#     # if (back < 0) or (back > 2):
#     #     return -np.inf
#     else:
#         return 0
    
# @nb.njit()
# def log_likelihood(theta, Kvals, Cmat):
#     """
#     1 species in 1 channel version
#     """
#     gamma2 = 0.01257
#     gamma3 = 0.00021
#     B1 = 5
#     B2 = 10
#     back = 1
#     N1 = theta[0]
#     N2 = theta[1]
#     cumulants = np.atleast_2d(Kvals)
#     if len(Cmat.shape) == 2:
#         arrC = np.reshape(Cmat, (1, Cmat.shape[0], Cmat.shape[1]))
#     else:
#         arrC = Cmat.copy()
#     pvals = np.zeros(len(cumulants))
#     k1 = 1*(N1*B1 + N2*B2) + back
#     k2 = gamma2*(N1*B1**2 + N2*B2**2)
#     k3 = gamma3*(N1*B1**3 + N2*B2**3)
#     model = np.array([k1, k2, k3])
#     for i in range(len(cumulants)):
#         pvals[i] = -0.5*( np.transpose(cumulants[i] - model) @ np.linalg.inv(arrC[i]) @ (cumulants[i] - model) ) - np.log(np.sqrt(np.abs(np.linalg.det(arrC[i])))*(2*np.pi)**(3/2))
#     return np.sum(pvals)

# @nb.njit()
# def log_posterior(theta, Kvals, Cmat):
    
#     logpost = flat_prior(theta) + log_likelihood(theta, Kvals, Cmat)  
#     return logpost 


#################################################################### maybe useful stuff below
"""
This section has the code for MCMC cumulant fitting which is a largely pointless endeavour :(
"""
# @nb.njit()
def ln_func1c(gamma1, gamma2, gamma3, N1, N2, N3, B1, B2, B3, back, cumulants, arrC):
    pvals = np.zeros(len(cumulants))
    k1 = gamma1*(N1*B1 + N2*B2 + N3*B3) + back
    k2 = gamma2*(N1*B1**2 + N2*B2**2 + N3*B3**2)
    k3 = gamma3*(N1*B1**3 + N2*B2**3 + N3*B3**3)
    model = np.array([k1, k2, k3])
    for i in range(len(cumulants)):
        pvals[i] = -0.5*( np.transpose(cumulants[i] - model) @ np.linalg.inv(arrC[i]) @ (cumulants[i] - model) ) - np.log(np.sqrt(np.abs(np.linalg.det(arrC[i])))*(2*np.pi)**(3/2))
    return np.sum(pvals)   

# @nb.njit()
def ln_func2c(gamma1, gamma2, gamma3, N, B, backa, backb, cumulants, arrC):
    # N = np.array([N1, N2, N3])
    # B = np.array([[B1a, B1b], [B2a, B2b], [B1a + B2a, B1b + B2b]], dtype='float64')
    pvals = np.zeros(len(cumulants))
    k10 = gamma1 * np.sum( N * B[:, 0]) + backa
    k01 = gamma1 * np.sum( N * B[:, 1]) + backb
    k20 = gamma2 * np.sum( N * B[:, 0]**2)
    k02 = gamma2 * np.sum( N * B[:, 1]**2)
    k11 = gamma2 * np.sum( N * B[:, 0] * B[:, 1] )
    k30 = gamma3 * np.sum( N * B[:, 0]**3)
    k03 = gamma3 * np.sum( N * B[:, 1]**3)
    k21 = gamma3 * np.sum( N * B[:, 0]**2 * B[:, 1])
    k12 = gamma3 * np.sum( N * B[:, 0] * B[:, 1]**2)
    model = np.array([k10, k01, k20, k02, k11, k30, k03, k21, k12])
    # print('C')
    # print(arrC.shape)
    # print(len(cumulants))
    for i in range(len(cumulants)):
        # print(np.linalg.inv(arrC[i]))
        # print(np.linalg.det(arrC[i]))
        pvals[i] = -0.5*( np.transpose(cumulants[i] - model) @ np.linalg.inv(arrC[i]) @ (cumulants[i] - model) ) - np.log(np.sqrt(np.abs(np.linalg.det(arrC[i])))*(2*np.pi)**(3/2))
    # if np.any(np.isnan(pvals)):
    #     print('here')
    #     print(pvals)
    #     print(model)
    #     print(N)
    #     print(B)
        # for i in range(len(cumulants)):
        #     print(np.linalg.inv(arrC[i]))
        #     print(np.linalg.det(arrC[i]))
    return np.sum(pvals)   

# @nb.njit()
def log_gaussian(x, mu, sigma):
    # ln2pi = 1.8378770664093453
    if x > 1e10:
        return -np.inf
    return (-0.5*(x-mu)**2 / (sigma**2)) - 0.5*(1.8378770664093453 + 2*np.log(sigma))

class mc_sampler():
    def __init__(self, theta_dict):
        self.gamma1 = 1
        self.gamma2 = theta_dict['gamma2'][0]
        self.gamma3 = theta_dict['gamma3'][0]
        self.sampler = None
        self.theta = None
        self.single_channel=True
        self.parse_theta(theta_dict)

    def parse_theta(self, theta_dict):
        params = theta_dict.keys()
        self.free_pars = {}
        self.fixed_pars = {}
        for par in params:
            if theta_dict[par][2] == 'fixed':
                self.fixed_pars[par] = theta_dict[par][0]
            elif theta_dict[par][2] in ('flat', 'gaussian'):
                self.free_pars[par] = (theta_dict[par], len(self.free_pars))
            else:
                raise ValueError('Unknown param prior category')
    
    def log_likelihood_1c(self, theta, cumulants, arrC):
        gamma1 = self.gamma1
        gamma2 = self.gamma2
        gamma3 = self.gamma3

        if 'N1' in self.fixed_pars:
            N1 = self.fixed_pars['N1']
        else:
            N1 = theta[self.free_pars['N1'][1]] 
        if 'N2' in self.fixed_pars:
            N2 = self.fixed_pars['N2']
        else:
            N2 = theta[self.free_pars['N2'][1]]
        if 'N3' in self.fixed_pars:
            N3 = self.fixed_pars['N3']
        else:
            N3 = theta[self.free_pars['N3'][1]]
        if 'B1' in self.fixed_pars:
            B1 = self.fixed_pars['B1']
        else:
            B1 = theta[self.free_pars['B1'][1]]
        if 'B2' in self.fixed_pars:
            B2 = self.fixed_pars['B2']
        else:
            B2 = theta[self.free_pars['B2'][1]]
        if 'B3' in self.fixed_pars:
            B3 = self.fixed_pars['B3']
        else:
            B3 = theta[self.free_pars['B3'][1]]
        if 'back' in self.fixed_pars:
            back = self.fixed_pars['back']
        else:
            back = theta[self.free_pars['back'][1]]

        # pvals = np.zeros(len(cumulants))
        # k1 = gamma1*(N1*B1 + N2*B2 + N3*B3) + back
        # k2 = gamma2*(N1*B1**2 + N2*B2**2 + N3*B3**2)
        # k3 = gamma3*(N1*B1**3 + N2*B2**3 + N3*B3**3)
        # model = np.array([k1, k2, k3])
        # for i in range(len(cumulants)):
        #     pvals[i] = -0.5*( np.transpose(cumulants[i] - model) @ np.linalg.inv(arrC[i]) @ (cumulants[i] - model) ) - np.log(np.sqrt(np.abs(np.linalg.det(arrC[i])))*(2*np.pi)**(3/2))
        # return np.sum(pvals)   

        ### This is a weird workaround for using numba in the class where I "outsource" the non-dictionary parsing calculations to a numba-decorated function that doesn't have to deal with dict types.
        return ln_func1c(gamma1, gamma2, gamma3, N1, N2, N3, B1, B2, B3, back, cumulants, arrC)
    
    def log_likelihood_2c(self, theta, cumulants, arrC):
        gamma1 = self.gamma1
        # gamma2 = self.gamma2
        # gamma3 = self.gamma3

        if 'gamma2' in self.fixed_pars:
            gamma2 = self.gamma2
        else:
            gamma2 = theta[self.free_pars['gamma2'][1]]
        if 'gamma3' in self.fixed_pars:
            gamma3 = self.gamma3
        else:
            gamma3 = theta[self.free_pars['gamma3'][1]]
        if 'N1' in self.fixed_pars:
            N1 = self.fixed_pars['N1']
        else:
            N1 = theta[self.free_pars['N1'][1]] 
        if 'N2' in self.fixed_pars:
            N2 = self.fixed_pars['N2']
        else:
            N2 = theta[self.free_pars['N2'][1]]
        if 'N3' in self.fixed_pars:
            N3 = self.fixed_pars['N3']
        else:
            N3 = theta[self.free_pars['N3'][1]]
        if 'B1a' in self.fixed_pars:
            B1a = self.fixed_pars['B1a']
        else:
            B1a = theta[self.free_pars['B1a'][1]]
        if 'B1b' in self.fixed_pars:
            B1b = self.fixed_pars['B1b']
        else:
            B1b = theta[self.free_pars['B1b'][1]]
        if 'B2a' in self.fixed_pars:
            B2a = self.fixed_pars['B2a']
        else:
            B2a = theta[self.free_pars['B2a'][1]]
        if 'B2b' in self.fixed_pars:
            B2b = self.fixed_pars['B2b']
        else:
            B2b = theta[self.free_pars['B2b'][1]]
        if 'backa' in self.fixed_pars:
            backa = self.fixed_pars['backa']
        else:
            backa = theta[self.free_pars['backa'][1]]
        if 'backb' in self.fixed_pars:
            backb = self.fixed_pars['backb']
        else:
            backb = theta[self.free_pars['backb'][1]]

        N = np.array([N1, N2, N3])
        B = np.array([[B1a, B1b], [B2a, B2b], [B1a + B2a, B1b + B2b]])

        # pvals = np.zeros(len(cumulants))
        # k10 = gamma1 * np.sum( N * B[:, 0]) + backa
        # k01 = gamma1 * np.sum( N * B[:, 1]) + backb
        # k20 = gamma2 * np.sum( N * B[:, 0]**2)
        # k02 = gamma2 * np.sum( N * B[:, 1]**2)
        # k11 = gamma2 * np.sum( N * B[:, 0] * B[:, 1] )
        # k30 = gamma3 * np.sum( N * B[:, 0]**3)
        # k03 = gamma3 * np.sum( N * B[:, 1]**3)
        # k21 = gamma3 * np.sum( N * B[:, 0]**2 * B[:, 1])
        # k12 = gamma3 * np.sum( N * B[:, 0] * B[:, 1]**2)
        # model = np.array([k10, k01, k20, k02, k11, k30, k03, k21, k12])
        # for i in range(len(cumulants)):
        #     pvals[i] = -0.5*( np.transpose(cumulants[i] - model) @ np.linalg.inv(arrC[i]) @ (cumulants[i] - model) ) - np.log(np.sqrt(np.abs(np.linalg.det(arrC[i])))*(2*np.pi)**(3/2))
        # return np.sum(pvals)   

        return ln_func2c(gamma1, gamma2, gamma3, N, B, backa, backb, cumulants, arrC)
    
    
    def log_prior(self, theta):
        params = self.free_pars.keys()
        logpri = 0
        for par in params:
            parval = theta[self.free_pars[par][1]]
            if parval < 0:
                return -np.inf
            if self.free_pars[par][0][2] == 'flat':
                if self.free_pars[par][0][0] <= parval < self.free_pars[par][0][1]:
                    logpri += np.log(1/(self.free_pars[par][0][1] - self.free_pars[par][0][0]))
                else:
                    return -np.inf
            else:
                logpri += log_gaussian(parval, self.free_pars[par][0][0], self.free_pars[par][0][1])
        return logpri

    def log_posterior_1c(self, theta, Kvals, Cmat):
        return self.log_prior(theta) + self.log_likelihood_1c(theta, Kvals, Cmat)
    
    def log_posterior_2c(self, theta, Kvals, Cmat):
        # pri = self.log_prior(theta)
        # like = self.log_likelihood_2c(theta, Kvals, Cmat)
        # if np.isnan(like):
        #     print('HERE in likelihood!')
        # if np.isnan(pri):
        #     print('in prior')
        return self.log_prior(theta) + self.log_likelihood_2c(theta, Kvals, Cmat) 

    # def make_sampler(self, nwalkers, cumulants, Cmat, single_channel=True):
    #     Kvals = np.atleast_2d(cumulants)
    #     if len(Cmat.shape) == 2:
    #         arrC = np.reshape(Cmat, (1, Cmat.shape[0], Cmat.shape[1]))
    #     else:
    #         arrC = Cmat.copy()
    #     if single_channel:
    #         self.sampler = mc.EnsembleSampler(nwalkers, len(self.free_pars), self.log_posterior_1c, args=[Kvals, arrC])
    #     else:
    #         self.sampler = mc.EnsembleSampler(nwalkers, len(self.free_pars), self.log_posterior_2c, args=[Kvals, arrC])

    def run_mcmc(self, starting_guess, nsteps, progress=True):
        self.sampler.run_mcmc(starting_guess, nsteps, progress=progress)