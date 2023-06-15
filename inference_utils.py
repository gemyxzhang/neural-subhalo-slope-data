import numpy as np
import os, sys
import torch 
from resnet import ResNetRatioEstimator


def to_tensor(x, device=torch.device('cuda:0')): 
    '''
    Args: 
        x (np.array, torch.Tensor): data being sent to gpu 
        device (optional, torch.device): default torch.device('cuda:0'); gpu device for sending x 

    Returns: 
        x (torch.Tensor)
    '''
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x).type(torch.float32).to(device)
    else:
        x = x.type(torch.float32).to(device)
        
    return x 

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def load_model(load_dir, epoch, p_dropout=0, device=torch.device('cuda:0'), mean=None, std=None):
    '''
    Args: 
        load_dir (str): path of model 
        epoch (int): epoch of trained model 
        device (optional, torch.device): default torch.device('cuda:0'); gpu device for sending x
        
    Returns: 
        ResNetRatioEstimator model loaded with load_state_dict
    '''
    checkpoint = torch.load(load_dir + 'epoch%s_checkpt.pth' % epoch)
    args = checkpoint['args']

    if (not hasattr(args, 'cfg')): 
        args.cfg = 18 
        
    print('cfg = {}'.format(args.cfg)) 
    model = ResNetRatioEstimator(cfg=args.cfg, n_aux=1, n_out=args.num_features, input_mean=mean, input_std=std, p_dropout=p_dropout).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def compute_logr(thetas_test, input_test, model, option='lld'): 
    '''
    Args: 
        thetas_test (np.array, torch.Tensor): needs shape (len(thetas_test), 1); array of parameters to be tested 
        input_test (np.array, torch.Tensor): len(thetas_test) = len(input_test)
        model (ResNetRatioEstimator): inference model 
        option (str): allowed options include 'lld' for likelihood logr, 
                    'ref' for reference logr, and 'both' for both logrs
                    
    Returns: 
        logrs: log likelihood-ratios (torch.Tensor of shape (len(logrs),)) from 
                putting input_test and thetas_test into model 
    '''
    
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
    '''
    Args: 
        thetas_test (np.array, torch.Tensor): array of parameters to be tested 
        data_test (np.array, torch.Tensor): the image to test thetas_test with 
        model (ResNetRatioEstimator): inference model 
        
    Returns: 
        logrs (np.array): array of llrs corresponding to theta_test array, with each column 
            being the llrs for each image (properly normalized) 
    '''
    
    if (np.isscalar(data_test)): data_test = [data_test]
        
    logrs = [] 
    
    for data in data_test: 
        logr = compute_logr(thetas_test.reshape((len(thetas_test), 1)), np.array([data]*len(thetas_test)), model)
        
        logrs.append(logr.cpu().detach().numpy())

    logrs = np.array(logrs).T
    logrs -= np.amax(logrs, axis=0).T

    return logrs 


    
def load_data(path_train, path_val, n_data, zero_nsub=False): 
    '''
    Args: 
        path_train (str): path to load training set info 
        path_val (str): path to load validation set info 
        n_data (int): number of data points in training set 
        zero_nsub (bool): default False; whether to replace the gamma values of inputs with nusb=0 with 0 
    
    Returns: 
        training set gammas (np.array), validation set gammas (np.array), training set gamma mean (float), 
        training set data mean (float), training set data std (float)
    '''
    
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