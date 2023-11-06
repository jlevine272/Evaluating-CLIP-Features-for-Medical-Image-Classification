from glob import glob
from PIL import Image
# import numpy as np
import pandas as pd
import os # , cv2,itertools
from torch.utils.data import Dataset, random_split
import torch

LESION_TYPE = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

def load_dataset(name, transform=None, data_dir=None):
    """
    Loads the specified dataset (either HAM10000 or NIH) from
    data_dir, applying transform.
    
    Returns a training and testing/val dataset
    """
    if name == "HAM10000":
        if data_dir is None:
            data_dir = "data/ham10000"
            
        return load_ham10000_dataset(data_dir, transform, True)
    elif name == "NIH":
        # TODO: write function to load NIH dataset. We probably need to modify this stuff a bit bc NIH give train/val/test but HAM is only one split.
        if data_dir is None:
            data_dir = "data/nih"
            
        raise NotImplementedError("This dataset isn't implemented")
    else:
        raise ValueError("expected either 'HAM10000' or 'NIH', but received " + name)

def load_ham10000_dataset(data_dir="data/ham10000", transform=None, split=True):
    print("Loading HAM10000 dataset...")
    df = get_dataframe(data_dir)
    dataset = HAM10000(df, transform)
    if split:
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset
    else:
        return dataset
    


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

def get_dataframe(data_dir):
    # https://www.kaggle.com/code/xinruizhuang/skin-lesion-classification-acc-90-pytorch
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(LESION_TYPE.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    return df_original