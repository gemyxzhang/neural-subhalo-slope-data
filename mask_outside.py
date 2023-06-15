import numpy as np
import time, sys, os
import argparse

parser = argparse.ArgumentParser('Mask images')

parser.add_argument("--n_data", type=int, default=500000, help='Number of samples to load.')
parser.add_argument("--deltapix", type=float, default=0.08, help='Pixel resolution.')
parser.add_argument("--numpix", type=int, default=100, help='Number of pixels per side of image.')
parser.add_argument("--n_start", type=int, default=0, help='Start index of images.')
parser.add_argument("--path", type=str, help='Path of data.')

args = parser.parse_args()

print('Masking', flush=True)
print(args.path, flush=True) 

x_grid, y_grid = np.meshgrid(np.arange(args.numpix), np.arange(args.numpix))
x_grid, y_grid = x_grid.reshape((args.numpix, args.numpix)), y_grid.reshape((args.numpix, args.numpix))

for i in range(args.n_start, args.n_start + args.n_data):
    im = np.load(args.path + 'images/SLimage_{}.npy'.format(i + 1))
    lensargs = np.load(args.path + 'lensargs/lensarg_{}.npy'.format(i + 1), allow_pickle=True)[0]
    mask_r = 1.5*lensargs['theta_E']/args.deltapix
    x_c, y_c = lensargs['center_x']/args.deltapix+args.numpix/2+0.5, lensargs['center_y']/args.deltapix+args.numpix/2+0.5
    
    mask = (x_grid - x_c)**2 + (y_grid - y_c)**2 < mask_r**2
    
    np.save(args.path + 'images/SLimage_maskedge_{}'.format(i + 1), mask*im)

    if ((i+1) % 5000 == 0): 
        print('Image {} saved'.format(i + 1), flush=True)