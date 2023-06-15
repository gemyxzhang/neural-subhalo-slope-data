import numpy as np
import time
import os, sys, argparse
import torch 

from scipy import interpolate
from scipy import stats
from scipy.stats import chi2, rv_histogram

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut, GridSearchCV

from resnet import ResNetRatioEstimator
from inference_utils import * 


parser = argparse.ArgumentParser('Calculate matrix of r that can approximate calibration distributions')
parser.add_argument("--n", type=int, default=100000, help='Number of images (for ref) or number of images per gamma bin.')
parser.add_argument("--n_bin", type=int, default=100000, help='Number of bins in gamma (for numerator distribution).')
parser.add_argument("--option", type=str)

args = parser.parse_args()

'''
def to_tensor(x, device=torch.device('cuda:0')): 
    
    Args: 
        x (np.array, torch.Tensor): data being sent to gpu 
        device (optional, torch.device): default torch.device('cuda:0'); gpu device for sending x 

    Returns: 
        x (torch.Tensor)
    
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x).type(torch.float32).to(device)
    else:
        x = x.type(torch.float32).to(device)
        
    return x 

def load_model(load_dir, epoch, p_dropout=0, device=torch.device('cuda:0'), mean=None, std=None):
    
    Args: 
        load_dir (str): path of model 
        epoch (int): epoch of trained model 
        device (optional, torch.device): default torch.device('cuda:0'); gpu device for sending x
        
    Returns: 
        ResNetRatioEstimator model loaded with load_state_dict
    
    checkpoint = torch.load(load_dir + 'epoch%s_checkpt.pth' % epoch)
    args = checkpoint['args']

    if (not hasattr(args, 'cfg')): 
        args.cfg = 18 
        
    print('cfg = {}'.format(args.cfg)) 
    model = ResNetRatioEstimator(cfg=args.cfg, n_aux=1, n_out=args.num_features, input_mean=mean, input_std=std, p_dropout=p_dropout).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def load_data(path_train, path_val, n_data, zero_nsub=False): 
    
    Args: 
        path_train (str): path to load training set info 
        path_val (str): path to load validation set info 
        n_data (int): number of data points in training set 
    
    Returns: 
        training set gammas (np.array), validation set gammas (np.array), training set gamma mean (float), 
        training set data mean (float), training set data std (float)
    
    
    gammas_train = np.load(path_train + 'gammas_all.npy')[:n_data]
    gammas_val = np.load(path_val + 'gammas_all.npy')
    
    if (zero_nsub): 
        nsubs = np.load(path_train + 'nsubs_all.npy')[:n_data]
        nsubs_val = np.load(path_val + 'nsubs_all.npy')
        gammas_train = np.where(nsubs > 0, gammas_train, 0)
        gammas_val = np.where(nsubs_val > 0, gammas_val, 0) 

    gammas_mean = np.mean(gammas_train, axis=0)

    data_mean = np.load(path_train + 'im_mean_{}.npy'.format(n_data))
    if (os.path.exists(path_train + 'im_std_{}.npy'.format(n_data))): 
        data_std = np.load(path_train + 'im_std_{}.npy'.format(n_data))
    else: 
        data_std = 0.3 
    
    return gammas_train, gammas_val, gammas_mean, data_mean, data_std 


def compute_logr(thetas_test, input_test, model, option='lld'): 

    Args: 
        thetas_test (np.array, torch.Tensor): needs shape (len(thetas_test), 1); array of parameters to be tested 
        input_test (np.array, torch.Tensor): len(thetas_test) = len(input_test)
        model (ResNetRatioEstimator): inference model 
        option (str): allowed options include 'lld' for likelihood logr, 
                    'ref' for reference logr, and 'both' for both logrs
                    
    Returns: 
        logrs: log likelihood-ratios (torch.Tensor of shape (len(logrs),)) from 
                putting input_test and thetas_test into model 
    
    
    assert option == 'lld' or option == 'ref' or option == 'both'
    
    with torch.no_grad():
        thetas_test = to_tensor(thetas_test)
        input_test = to_tensor(input_test)
        
        if (option == 'lld'): 
            _, logrs = model(input_test, x_aux=thetas_test, train=False)
        elif (option == 'ref'):
            _, logrs = model(input_test, x_aux=thetas_test, train=True)
            logrs = logrs[1::2]
        else: 
            _, logrs = model(input_test, x_aux=thetas_test, train=True)
        
    return logrs.flatten()


def get_logrs(thetas_test, data_test, model):
    
    Args: 
        thetas_test (np.array, torch.Tensor): array of parameters to be tested 
        data_test (np.array, torch.Tensor): the image to test thetas_test with 
        model (ResNetRatioEstimator): inference model 
        
    Returns: 
        logrs (np.array): array of llrs corresponding to theta_test array, with each column 
            being the llrs for each image 
    
    
    if (np.isscalar(data_test)): data_test = [data_test]
        
    logrs = [] 
    
    for data in data_test: 
        logr = compute_logr(thetas_test.reshape((len(thetas_test), 1)), np.array([data]*len(thetas_test)), model)
        
        logrs.append(logr.cpu().detach().numpy())

    logrs = np.array(logrs).T
    logrs -= np.amax(logrs, axis=0).T

    return logrs 


'''

gammas_test = np.linspace(1.1, 2.9, args.n_bin)
gammas_test_ctrs = 0.5*(gammas_test[:-1] + gammas_test[1:])

PATH = '/n/holyscratch01/dvorkin_lab/gzhang/Storage/llr_data_images/F814W_pmod_deltapix0.04_numpix100_EPLsh_EPLml_logm7.0to10.0_beta-1.9_nsub0to500_0.5maxpix_g1.1to2.9_gammaw0.1_zl0.2zs0.6_shear0.1_nms_exptime2200_lenslight_multipole_los/'

PATH_calib = PATH + 'val/F814W_pmod_deltapix0.04_numpix100_EPLsh_EPLml_logm7.0to10.0_beta-1.9_nsub0to300_0.5maxpix_g1.1to2.9_gammaw0.1_zl0.2zs0.6_shear0.1_nms_exptime2200_lenslight_multipole_los/' 

PATH_model = PATH + 'models/maskedge_n32_resnet50_AdamW_dout0.0_lr0.001_bs1000_ndata5000000/'

PATH_save = PATH_model + 'calib/'
os.makedirs(PATH_save, exist_ok=True)

gs_train, gammas_calib, g_mean, mean, std = load_data(PATH, PATH_calib, 5000000, zero_nsub=False) 
model = load_model(PATH_model + 'arrays/', epoch=25)#, mean=to_tensor(mean), std=to_tensor(std))
model.eval()


# for the reference distribution 
if (args.option == 'ref'): 
    start_time = time.time()

    ims_calib = []

    for i in range(args.n): 
        im = np.load(PATH_calib + 'images/SLimage_maskedge_{}.npy'.format(i+1))
        ims_calib.append(im)

    ims_calib = np.array(ims_calib) 
    ma = (ims_calib != 0)
    ims_calib = ma*(ims_calib - mean)/std

    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    # get r of images from model 
    logrs_ref = get_logrs(gammas_test_ctrs - g_mean, ims_calib, model)

    assert np.max(logrs_ref.T[0]) == 0

    np.save(PATH_save + 'logrs_ref_n{}_nbin{}'.format(args.n, args.n_bin), logrs_ref)
    
    print("--- %s seconds ---" % (time.time() - start_time))

# for the target distribution 
else: 
    start_time = time.time()

    # each row correspond to distribution of logr for a gamma in gammas_test 
    logrs_calib = []

    # loop through gammas 
    for i, (g, g_next) in enumerate(zip(gammas_test[:-1], gammas_test[1:])): 
        # get calibration images at each gamma 
        inds_g = np.where((gammas_calib > g)*(gammas_calib < g_next))[0]

        ims_g = [] 
        for j in inds_g[:args.n]: 
            im = np.load(PATH_calib + 'images/SLimage_maskedge_{}.npy'.format(j+1))
            ims_g.append(im)

        ims_g = np.array(ims_g) 
        ma = (ims_g != 0)
        ims_g = ma*(ims_g - mean)/std
        
        # get r of images from model 
        logrs_temp = get_logrs(gammas_test_ctrs - g_mean, ims_g, model)

        assert np.max(logrs_temp.T[0]) == 0

        logrs_calib.append(logrs_temp[i])

    np.save(PATH_save + 'logrs_calib_n{}_nbin{}'.format(args.n, args.n_bin), logrs_calib)
    

    print("--- %s seconds ---" % (time.time() - start_time))