from glob import glob
import random
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

# here are 15 classes (14 diseases, and one for "No findings"). 
# Images can be classified as "No findings" or one or more disease classes:
NIH_CLASS_TYPES = {
        'Atelectasis'           : 'Atelectasis',
        'Consolidation'         : 'Consolidation',
        'Infiltration'          : 'Infiltration',
        'Pneumothorax'          : 'Pneumothorax',
        'Edema'                 : 'Edema',
        'Emphysema'             : 'Emphysema',
        'Fibrosis'              : 'Fibrosis',
        'Effusion'              : 'Effusion',
        'Pneumonia'             : 'Pneumonia',
        'Pleural_thickening'    : 'Pleural_thickening',
        'Cardiomegaly'          : 'Cardiomegaly',
        'Nodule'                : 'Nodule',
        'Mass'                  : 'Mass',
        'Hernia'                : 'Hernia',
        'No Finding'            : 'No Finding'
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
            
        return load_nih_dataset_split(data_dir, transform)
    else:
        raise ValueError("expected either 'HAM10000' or 'NIH', but received " + name)

def load_ham10000_dataset(data_dir="data/ham10000/", transform=None, split=True):
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

class NIHDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        print('NIH Dataset: ', self.transform)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Load the image
        X = Image.open(self.dataframe['path'][index])

        # Convert the image to RGB
        X = X.convert('RGB')

        # Apply other transformations if necessary
        if self.transform:
            X = self.transform(X)

        # Load the label
        y = self.dataframe['cell_type_idx'][index]

        return X, y

def get_dataframes(root_dir, csv_file):
    # print("Getting root...", root_dir)
    # print("Loading NIH dataset...", csv_file)
    all_image_path = glob(os.path.join(root_dir, '*.png'))
    # print('all_image_path: ', all_image_path)

    # Ensure the keys include the file extension to match the 'Image Index' in the CSV
    image_id_path_dict = {os.path.splitext(os.path.basename(x))[0] + '.png': x for x in all_image_path}
    # print('image_id_path_dict: ', image_id_path_dict)
    df_original = pd.read_csv(csv_file)
    # print('df_original: ', df_original)
    # Debugging: Print a few values from the CSV to check their format
    # print('CSV Image Index sample:', df_original['Image Index'].head())

    df_original['path'] = df_original['Image Index'].map(image_id_path_dict.get)

    df_original['cell_type'] = df_original['Finding Labels'].map(NIH_CLASS_TYPES.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    return df_original


def load_nih_dataset_split(data_dir="data/nih/", transform=None, split=True):
    """
    Loads the NIH dataset from data_dir, applying transform.

    Returns a training and testing/val dataset
    """
    # load csv file
    csv_file_location = os.path.join(data_dir, "Data_Entry_2017.csv")
    # print(f"Loading csv file from {csv_file_location}")

    # load dataset
    # path ex: data\nih\images\00001336_000.png
    dataset_location = os.path.join(data_dir, "images")
    # print(f"Loading dataset from {dataset_location}")
    df = get_dataframes(dataset_location, csv_file_location)
    # print(df.head())
    dataset = NIHDataset(df, transform)
    # print(f"Original Dataset length: {dataset.__len__()}")
    # print(f"Original Dataset sample: {dataset.__getitem__(0)}")

    if split:
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset
    else:
        return dataset