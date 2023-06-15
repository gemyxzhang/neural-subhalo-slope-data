import numpy as np
import torch
from torch.utils.data import Dataset

    
# dataset Class 
class LensingDataset(Dataset):
    """Lensing images dataset."""
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (np.array): data being used 
            labels (np.array): conditionals  
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.data = data 
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        '''
        returns the idxth image in data as a tensor 
        '''
        sample = torch.tensor([self.data[idx]])   # extra [] for the shape dimensions to match for labels and sample for MAF
        label = torch.tensor(self.labels[idx])
        
        if self.transform:
            sample = self.transform(sample)      
            
        return sample, label 
    
    
    
class DatasetMixed(Dataset):
    """ Load folder of npy files
    """
    def __init__(self, root, params, num_files):
        '''
        Args: 
            root (str): path to load image files 
            params (np.array): param of interest matching those loaded from root 
            num_files (int): number of files to be loaded from root 
        '''
        self.root = root
        self.num_files = num_files
        self.params = params 

    def __getitem__(self, index):
        """ Load tuple of image, params
        """

        image = torch.tensor(np.load(self.root + "SLimage_{}.npy".format(index + 1)))
        param = torch.tensor([self.params[index]])

        return param, image

    def __len__(self):
        return self.num_files
    

class DatasetMixedMasked(Dataset):
    """ Load folder of npy files
    """
    def __init__(self, root, params, num_files):
        '''
        Args: 
            root (str): path to load image files 
            params (np.array): param of interest matching those loaded from root 
            num_files (int): number of files to be loaded from root 
        '''
        self.root = root
        self.num_files = num_files
        self.params = params 

    def __getitem__(self, index):
        """ Load tuple of image, params
        """

        image = torch.tensor(np.load(self.root + "SLimage_maskedge_{}.npy".format(index + 1)))
        param = torch.tensor([self.params[index]])

        return param, image

    def __len__(self):
        return self.num_files
    
    
class DatasetMixedMaskedMultifolder(Dataset):
    """ Load folder of npy files
    """
    def __init__(self, roots, nums_file, params, num_files):
        '''
        Args: 
            root (str): path to load image files 
            params (np.array): param of interest matching those loaded from root 
            num_files (int): number of files to be loaded from root 
        '''
        self.roots = roots
        self.n_total = np.sum(nums_file) 
        self.nums_file = np.cumsum(nums_file) 
        self.params = params 

    def __getitem__(self, index):
        """ Load tuple of image, params
        """
        idx_root = np.searchsorted(self.nums_file, index + 1)
        root = self.roots[idx_root]

        image = torch.tensor(np.load(root + "SLimage_maskedge_{}.npy".format(index + 1)))
        param = torch.tensor([self.params[index]])

        return param, image

    def __len__(self):
        return self.n_total 


class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support 
    """

    def __init__(self, *arrays, dtype=torch.float):
        self.dtype = dtype
        self.memmap = []
        self.data = []
        self.n = None

        for array in arrays:
            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n
