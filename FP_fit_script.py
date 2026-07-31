import numpy as np              
import bayes_fit as bf          
import glob     
import pandas as pd



def fit_FP_model(files, savepath, name_pos):
    res_dict = {'file name':[],
                'full_data_KD':[],
                'full_95_min':[],
                'full_95_max':[],
                'KD_range_min':[],
                'KD_range_max':[],    
                'extrema':[]
    }
    for file in files:
        print()
        print(file[name_pos:])
        f = np.load(file)
        Rvals = f['Rvals']
        Gvals = f['Gvals']
        RGvals = f['RGvals']
        yfpvals = f['yfpvals']
        mchvals = f['mchvals']
        Npixels = f['Npixels']
        where = np.ones(Rvals.shape, dtype='bool')
        where[Npixels < 0.8*np.max(Npixels)] = 0
        Ruse = Rvals[where]
        Guse = Gvals[where]
        RGuse = RGvals[where]

        N_R = np.mean(Ruse+RGuse)*44 # scale 150 nm pixels size to um^-2 concentrations.
        N_G = np.mean(Guse+RGuse)*44
        if not N_R > N_G: # "actual" identity of R and G doesnt matter as long as its a 2 species complex reaction
            N_R, N_G = N_G, N_R
            x_fits = RGuse/(RGuse + Ruse)
        else:
            x_fits = RGuse/(RGuse + Guse)
        xvals, xbins = np.histogram(x_fits, np.linspace(0, 1, int(N_G)//2), density=True)
        xpos = xpos = xbins[:-1] + np.diff(xbins)/2
        minseed, maxseed = bf.get_seed(N_G, N_R, offset=0, maxval=True)
        KD_grid = np.linspace(minseed, maxseed, max(int(maxseed-minseed)*2, 1000))
        log_likelihood, KD_grid = bf.log_likelihood_KD(x_fits, KD_grid, N_G, N_R, maxval=0)
        log_likelihood[np.logical_not(np.isfinite(log_likelihood))] = -np.inf
        log_likelihood = log_likelihood - np.max(log_likelihood)
        posterior = np.exp(log_likelihood) / (np.sum(np.exp(log_likelihood))*np.diff(KD_grid)[0])
        # zoom in once
        try:
            bounds = bf.get_confidence_interval(posterior, KD_grid, 1-1e-10)
        except Exception as e:
            print('zoom', e)
            bounds = [minseed, maxseed]
        new_grid = np.linspace(bounds[0], bounds[1], max(len(KD_grid), 2000))
        log_likelihood, new_grid = bf.log_likelihood_KD(x_fits, new_grid, N_G, N_R, maxval=0)
        log_likelihood[np.logical_not(np.isfinite(log_likelihood))] = -np.inf
        log_likelihood = log_likelihood - np.max(log_likelihood)
        posterior = np.exp(log_likelihood) / (np.sum(np.exp(log_likelihood))*np.diff(new_grid)[0])
        MAPKD = new_grid[np.argmax(log_likelihood)]
        try:
            err_bounds = bf.get_confidence_interval(posterior, new_grid, 0.95)
        except Exception as e:
            print('95% interval', e)
            err_bounds = bounds
        res_dict['file name'].append(file[name_pos:])
        res_dict['full_data_KD'].append(MAPKD)
        res_dict['full_95_min'].append(err_bounds[0])
        res_dict['full_95_max'].append(err_bounds[1])
        res_dict['KD_range_min'].append(minseed)
        res_dict['KD_range_max'].append(maxseed)
        if err_bounds[0] == bounds[0] or err_bounds[1] == bounds[1]:
            res_dict['extrema'].append(1)
        else:
            res_dict['extrema'].append(0)
    for key in res_dict.keys():
        print(key, len(res_dict[key]))
    df = pd.DataFrame(res_dict)
    df.to_excel(savepath + 'full_trace_KD_fit_results.xlsx')


if __name__ == "__main__":
    cumulant_path = input('Cumulant dir (will search for .npz files) : ')
    cumulant_pattern = cumulant_path + r'\\*.npz'
    files = glob.glob(cumulant_pattern)
    print(files)
    print(f'\nFound {len(files)} files to fit\n')    
    savepath = input('results dir (default is same as prev.) : ')
    if savepath == '':
        savepath = cumulant_path
    name_pos = len(cumulant_path) + 1
    fit_FP_model(files, savepath, name_pos)
