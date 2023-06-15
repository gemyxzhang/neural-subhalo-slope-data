###################################
# Train script 
###################################

import argparse
import os,sys
import time
from os.path import exists

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as torchdata 
from torch import nn

from resnet import ResNetRatioEstimator
from data_utils import * 


parser = argparse.ArgumentParser('Gamma Resnet')
parser.add_argument("--n_data", type=int, default=100000, help='Number of samples to load.')
parser.add_argument("--n_val", type=int, default=10000, help='Number of samples to load.')
parser.add_argument("--num_features", type=int, default=1, help='Number of features being studied.')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--transfer', action='store_true')
parser.add_argument('--reset_lr', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--nepochs', type=int, default=75)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.0) 
parser.add_argument('--cfg', type=int, default=18)
parser.add_argument('--n_worker', type=int, default=16)

parser.add_argument('--label', type=str)
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--path_data', type=str, default=None, help='Path to train data.')
parser.add_argument('--path_val', type=str, default=None)
parser.add_argument('--subidx_file', type=str, default=None)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--zero_nsub', action='store_true', help='Whether to account for 0 nsub.') 

args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.device = device 

def count_parameters(model):
    '''
    Args: 
        model: NN in pytorch 
        
    Returns: 
        number of params in model 
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def compute_loss(image, theta, loss_fn, model, device='cuda:0'): 
    '''
    Args: 
        image (np.array, torch.Tensor): image input into model 
        theta (np.array, torch.Tensor): parameter of interesting corresponding to image 
        loss_fn: loss function predefined in torch 
        model (ResNetRatioEstimator)
        device (optional, torch.device): default torch.device('cuda:0'); gpu device for sending x
    
    Returns: 
        loss from inputing image and theta into model 
    '''
    image = to_tensor(image, device) 
    theta = to_tensor(theta, device)

    batch_size = theta.shape[0]
    
    output,_ = model(image, x_aux=theta)
    
    # make the target classification labels 
    labels = torch.ones(2*batch_size).type_as(output)  # two atoms
    labels[1::2] = 0.0
        
    loss = loss_fn(torch.reshape(output, labels.size()), labels) 
    
    return loss


def compute_test_loss(image, theta, loss_fn, model, device='cuda:0'): 
    '''
    Args: 
        image (np.array, torch.Tensor): image input into model 
        theta (np.array, torch.Tensor): parameter of interesting corresponding to image 
        loss_fn: loss function predefined in torch 
        model (ResNetRatioEstimator)
        device (optional, torch.device): default torch.device('cuda:0'); gpu device for sending x
    
    Returns: 
        loss from inputing image and theta into model 
    '''
    with torch.no_grad(): 
        image = to_tensor(image, device) 
        theta = to_tensor(theta, device)

        batch_size = theta.shape[0]
        
        output,_ = model(image, x_aux=theta)
        
        # make the target classification labels 
        labels = torch.ones(2*batch_size).type_as(output)  # two atoms
        labels[1::2] = 0.0

        loss = loss_fn(torch.reshape(output, labels.size()), labels) 
    
    return loss

    
    
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
        
##################################################################################################

PATH_data = args.path_data
PATH_val = PATH_data + args.path_val
n_data = args.n_data
n_val = args.n_val 
n_train = n_data

# make or load directories

rootdir = PATH_data + 'models/'

if args.resume:
    print('Loading saved info!', flush=True)
    print('PATH: ' + args.load_dir, flush=True) 
    
    load_arrays = args.load_dir + 'arrays/'
    save_figures = args.load_dir + 'plots/' 
    save_arrays = args.load_dir + 'arrays/'
    
    ini, fin = args.epoch + 1, args.epoch + args.nepochs + 1
    
    print('Retrieving loss curves')
    
    train_losses = np.load(load_arrays + 'train_losses.npy').tolist()
    val_losses = np.load(load_arrays + 'val_losses.npy').tolist()

    train_losses = train_losses[:int(args.epoch*n_train/args.batch_size)]
    val_losses = val_losses[:int(args.epoch*n_train/args.batch_size)]

    
elif args.transfer: 
    print('Loading saved info and transferring!', flush=True)
    print('Loading from PATH: ' + args.load_dir, flush=True) 
    
    save_dir = rootdir + 'transfer_%s_resnet%s_%s_dout%s_lr%s_bs%s_ndata%s/' % (args.label, args.cfg, args.optimizer, args.dropout, args.lr, args.batch_size, args.n_data) 
    os.makedirs(save_dir, exist_ok=True) 
    print(save_dir, flush=True)
    
    save_figures = save_dir + 'plots/'
    os.makedirs(save_figures, exist_ok=True)

    save_arrays = save_dir + 'arrays/'
    os.makedirs(save_arrays,exist_ok=True)
    
    load_arrays = args.load_dir + 'arrays/'
    
    ini, fin = args.epoch + 1, args.epoch + args.nepochs + 1
    
    print('Retrieving loss curves')
    
    train_losses = np.load(load_arrays + 'train_losses.npy').tolist()
    val_losses = np.load(load_arrays + 'val_losses.npy').tolist()

    train_losses = train_losses[:int(args.epoch*n_train/args.batch_size)]
    val_losses = val_losses[:int(args.epoch*n_train/args.batch_size)]
    
else: 
    
    save_dir = rootdir + '%s_resnet%s_%s_dout%s_lr%s_bs%s_ndata%s/' % (args.label, args.cfg, args.optimizer, args.dropout, args.lr, args.batch_size, args.n_data) 
    os.makedirs(save_dir, exist_ok=True) 
    print(save_dir, flush=True)
    
    save_figures = save_dir + 'plots/'
    os.makedirs(save_figures, exist_ok=True)

    save_arrays = save_dir + 'arrays/'
    os.makedirs(save_arrays,exist_ok=True)
    
    ini, fin = 1, args.nepochs + 1

    train_losses, val_losses = [],[]
    
    
    
################################################################################

    
print('Loading train images', flush=True)
print(PATH_data, flush=True)

# load in parameters 
gammas = np.load(PATH_data + 'gammas_all.npy')[:args.n_data]
gammas_val = np.load(PATH_val + 'gammas_all.npy')[:args.n_val]

# set gamma of images with nsub=0 to 0 
if (args.zero_nsub):
    nsubs = np.load(PATH_data + 'nsubs_all.npy')[:args.n_data]
    nsubs_val = np.load(PATH_val + 'nsubs_all.npy')[:args.n_val]
    gammas = np.where(nsubs > 0, gammas, 0)
    gammas_val = np.where(nsubs_val > 0, gammas_val, 0) 

    
if args.transfer or args.resume: 
    mean = np.load(load_arrays + 'mean.npy') 
    
    if (exists(load_arrays + 'std.npy')): 
        std = np.load(load_arrays + 'std.npy') 
    else: 
        std = 0.3 
    
else: 
    # load in pre computed mean and std of images 
    mean = np.load(PATH_data + 'im_mean_{}.npy'.format(args.n_data)) 
    if (exists(PATH_data + 'im_std_{}.npy'.format(args.n_data))): 
        std = np.load(PATH_data + 'im_std_{}.npy'.format(args.n_data)) 
    else: 
        std = 0.3 
        
    np.save(save_arrays + 'mean', mean)
    np.save(save_arrays + 'std', std) 

    
#mean = to_tensor(mean, device)
#std = to_tensor(std, device) 

print(mean, flush=True)
print(std, flush=True) 

# load validation set 
print('Loading validation images', flush=True)
conditional = [] 

for i in range(args.n_val): 
    im = np.load(PATH_val + 'images/SLimage_maskedge_{}.npy'.format(i+1))
    conditional.append(im)

print('Shape of conditional: {}'.format(np.shape(conditional)), flush=True) 

# whiten parameters 
thetas = np.append(gammas, gammas_val) - np.mean(gammas, axis=0)

print('Shape of data: {}'.format(np.shape(thetas)), flush=True) 

# make train+val sets 

train_set = DatasetMixedMasked(PATH_data + 'images/', thetas[:n_train], n_train)
val_set = LensingDataset(thetas[n_train:], conditional)
trainloader = torchdata.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.n_worker)
valloader = torchdata.DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False)

print('Number of batches: {}'.format(len(trainloader)))
    
    


#################################################################################

# initialize model 

model = ResNetRatioEstimator(cfg=args.cfg, n_aux=1, n_out=args.num_features, p_dropout=args.dropout).to(device)
if (args.optimizer == 'AdamW'):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif (args.optimizer == 'SGD'):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
elif (args.optimizer == 'Adam'): 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.01, threshold_mode='abs', cooldown=2, verbose=True)

loss_fn = nn.BCELoss()

# load checkpoint if resuming training 
if args.resume or args.transfer:
    print('Loading checkpoint!', flush=True)
    
    checkpoints = torch.load(load_arrays + 'epoch%s_checkpt.pth' % args.epoch)
    model.load_state_dict(checkpoints['state_dict'])
    
    if args.resume: 
        args = checkpoints['args']
        
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    
    if args.transfer and not args.reset_lr: 
        args = checkpoints['args']
        
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])

print(model)
print(count_parameters(model))

##################################################################################

# for early stopping 
loss_bestepoch = np.inf  
count_earlystop = 0

# train loop 
for epoch in range(ini, fin):
    
    print('\n Epoch %s' % epoch, flush=True)

    # the sum of validation loss of all batches in a epoch 
    loss_total = 0. 

    for count, x in enumerate(trainloader):

        model.train()
        
        theta, x = x 
        mask = (x != 0)
        x = mask*(x - mean)/std  # whiten data 
        
        optimizer.zero_grad()
        loss = compute_loss(x, theta, loss_fn, model, device)

        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
    
        model.eval() 
        if (args.dropout): enable_dropout(model) 
        # average over the # of batches 
        val_loss = 0.
        for count_val, y in enumerate(valloader): 
            theta, y = y
            mask = (y != 0)
            y = mask*(y - mean)/std 
            val_loss += compute_test_loss(y, theta, loss_fn, model, device)
            
        val_losses.append(val_loss.item()/len(valloader))
        
        loss_total += val_loss.item()/len(valloader)

    loss_epoch = loss_total/len(trainloader)
    print('Val loss: {}'.format(loss_epoch), flush=True)
    
    scheduler.step(loss_epoch)
    print('lr: {}'.format(optimizer.param_groups[0]['lr']), flush=True)

    np.save(save_arrays + 'train_losses', train_losses)   
    np.save(save_arrays + 'val_losses', val_losses)
    
    torch.save({'args': args,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict()}, os.path.join(save_arrays, 'epoch%s_checkpt.pth' % epoch))
    
    
    # early stopping 
    if loss_bestepoch - loss_epoch < 0.001: 
        count_earlystop += 1 
    else: 
        loss_bestepoch = loss_epoch 
        count_earlystop = 0

    if count_earlystop == 3: break 