# Subhalo effective density slope measurements from HST strong lensing data with neural likelihood-ratio estimation
[![arXiv](https://img.shields.io/badge/arXiv-2208.13796%20-green.svg)](https://arxiv.org/abs/2308.09739)

## Software dependencies
The code uses standard `astropy`, `numpy`, `scipy` and `scikit-learn` packages. We use [paltas](https://github.com/swagnercarena/paltas) and [lenstronomy](https://github.com/lenstronomy/lenstronomy) for data generation. These can be installed as follows: 

```
pip install paltas lenstronomy 
```
The `tqdm` package is needed for the progress bar display but can be commented out easily. We ran our analysis with `Python 3.9.12` and `Pytorch 1.12.1`. 

## Architecture 
<p float="left">
  <img src="/model.png" width="85%" />
</p>

## Inference pipeline 

### Training set generation 
To generate mock lensing images, run the following script (which has dependency on [utils.py](utils.py)): 
- [make_images.py](make_images.py) makes lensing images with EPL subhalos for training and validation and NFW for testing 

Note: if this is run with slurm job arrays, you need to manually combine the gamma parameter files produced by the job arrays and determine the mean and standard deviation used for whitening.

To mask the edges of the mock images, run the following script:
- [mask_outside.py](mask_outside.py)

### Training 
To train a likelihood-ratio estimator on the mock images, run the following script. The likelihood-ratio estimator model class is in [resnet.py](resnet.py).
- [train_masked.py](train_masked.py) with the specified parameters (which has dependecy on [data_utils.py](data_utils.py))

### Inference 
If calibration is needed, to obtain the likelihood ratios for estimating the calibration distributions, the following script will produce two sets of likelihood ratios that can be used for KDE: 
- [make_calibration_distributions.py](make_calibration_distributions.py) (which has dependecy on [inference_utils.py](inference_utils.py))

The HST images used in our analysis are taken from 
- [https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)

Finally, 
- [figures.ipynb](figures.ipynb) contains code that produces the main results
