import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SCAPE(Dataset):
    '''
   
    '''
    def __init__(self, data_path, transform=None):
        '''
        Args:
            data_path (str): path to dataset
        '''
        self.data_path = data_path
        self.transform = transform
        self.fpaths = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
        gray_lst = ['00000356_(5)','00000172_(6)','00000692_(2)']
        for num in gray_lst:
            self.fpaths.remove(os.path.join(data_path, '{}.jpg'.format(num)))

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.fpaths[idx]))
        return img

    def __len__(self):
        return len(self.fpaths)