# New Logic Sohaib


import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from flamby.utils import check_dataset_from_config


class HeartDiseaseRaw(Dataset):
    """Pytorch dataset containing all the features, labels and
    metadata for the Heart Disease dataset.

    Parameters
    ----------
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    debug : bool, optional,
        Whether or not to use only the part of the dataset downloaded in
        debug mode. Defaults to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.

    Attributes
    ----------
    data_dir: str
        Where data files are located
    labels : pd.DataFrame
        The labels as a dataframe.
    features: pd.DataFrame
        The features as a dataframe.
    centers: list[int]
        The list with the center ids associated with the dataframes.
    sets: list[str]
        For each sample if it is from the train or the test.
    X_dtype: torch.dtype
        The dtype of the X features output
    y_dtype: torch.dtype
        The dtype of the y label output
    debug: bool
        Whether or not we use the dataset with only part of the features
    normalize: bool
        Whether or not to normalize the features. We use the corresponding
        training client to compute the mean and std per feature used to
        normalize.
        Defaults to True.
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        debug=False,
        data_path=None,
        normalize=True,
    ):
        """See description above"""

        directory = "/panfs/panfs.ittc.ku.edu/scratch/sanaawan/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset/outputt_csv/"
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug

        #self.centers_number = {"cleveland": 0, "hungarian": 1, "switzerland": 2, "va": 3}
        self.centers_number = {}

        # initialize a counter for the values in the dictionary
        value_counter = 0

        # iterate over the files in the directory and add them to the dictionary
        for filename in os.listdir(directory):
        # add the file name to the dictionary with the current value counter value
            filename_split = filename.split(".")[0]
            self.centers_number[filename_split] = value_counter

        # increment the value counter
            value_counter += 1


        #print(self.centers_number)
        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.centers = []
        self.sets = []
        self.removed_hospitalid=[]
        self.train_fraction = 0.66
        self.dir = Path(directory)
        for center_data_file in self.dir.glob("*.csv"):

            center_name = os.path.basename(center_data_file).split(".")[0]
            if not center_name:
                break
            df = pd.read_csv(center_data_file)
            df.drop(['patientunitstayid'],axis=1,inplace=True)

            #print ("data shape",df.shape)
            col_at_end = ['AKI_label']
            df = df[[c for c in df if c not in col_at_end]+[c for c in col_at_end if c in df]]


            center_X = df.iloc[:, :-1]
            center_y = df.iloc[:, -1]

            if center_X.shape[0] <= 500:
                self.removed_hospitalid.append(self.centers_number[center_name])
                #print ("Center in remove list")
                #del self.centers_number[center_name]
                continue
            print (center_name,center_X.shape)
            self.features = pd.concat((self.features, center_X), ignore_index=True)

            self.labels = pd.concat((self.labels, center_y), ignore_index=True)
            #print (self.features.shape,self.labels.shape)
            self.centers += [self.centers_number[center_name]] * center_X.shape[0]

            # proposed modification to introduce shuffling before splitting the center
            nb = int(center_X.shape[0])
            current_labels = center_y.where(center_y == 0, 1, inplace=False)
            levels = np.unique(current_labels)
            if (len(np.unique(current_labels)) > 1) and all(
                [(current_labels == lev).sum() > 2 for lev in levels]
            ):
                stratify = current_labels
            else:
                stratify = None
            indices_train, indices_test = train_test_split(
                np.arange(nb),
                test_size=1.0 - self.train_fraction,
                train_size=self.train_fraction,
                random_state=43,
                shuffle=True,
                stratify=stratify,
            )

            for i in np.arange(nb):
                if i in indices_test:
                    self.sets.append("test")
                else:
                    self.sets.append("train")

        # encode dummy variables for categorical variables
        #self.features = pd.get_dummies(self.features, columns=[2, 6], drop_first=True)
        self.features = [
            torch.from_numpy(self.features.loc[i].values.astype(np.float32)).to(
                self.X_dtype
            )
            for i in range(len(self.features))
        ]

        # keep 0 (no disease) and put 1 for all other values (disease)
        self.labels.where(self.labels == 0, 1, inplace=True)
        self.labels = torch.from_numpy(self.labels.values).to(self.X_dtype)

        # Per-center Normalization much needed
        self.centers_stats = {}
        self.centerr=[num for num in range(188)]
        for x in self.removed_hospitalid:
           self.centerr.remove(x)

        for center in self.centerr:
            # We normalize on train only
            to_select = [
                (self.sets[idx] == "train") and (self.centers[idx] == center)
                for idx, _ in enumerate(self.features)
            ]
            features_center = [
                fp for idx, fp in enumerate(self.features) if to_select[idx]
            ]
            features_tensor_center = torch.cat(
                [features_center[i][None, :] for i in range(len(features_center))],
                axis=0,
            )
            mean_of_features_center = features_tensor_center.mean(axis=0)
            std_of_features_center = features_tensor_center.std(axis=0)
            self.centers_stats[center] = {
                "mean": mean_of_features_center,
                "std": std_of_features_center,
            }

        # We finally broadcast the means and stds over all datasets
        self.mean_of_features = torch.zeros((len(self.features), 555), dtype=self.X_dtype)
        self.std_of_features = torch.ones((len(self.features), 555), dtype=self.X_dtype)
        for i in range(self.mean_of_features.shape[0]):
            self.mean_of_features[i] = self.centers_stats[self.centers[i]]["mean"]
            self.std_of_features[i] = self.centers_stats[self.centers[i]]["std"]

        # We normalize on train only for pooled as well
        to_select = [(self.sets[idx] == "train") for idx, _ in enumerate(self.features)]
        features_train = [fp for idx, fp in enumerate(self.features) if to_select[idx]]
        features_tensor_train = torch.cat(
            [features_train[i][None, :] for i in range(len(features_train))], axis=0
        )
        self.mean_of_features_pooled_train = features_tensor_train.mean(axis=0)
        self.std_of_features_pooled_train = features_tensor_train.std(axis=0)

        # We convert everything back into lists

        self.mean_of_features = torch.split(self.mean_of_features, 1)
        self.std_of_features = torch.split(self.std_of_features, 1)
        self.mean_of_features_pooled_train = [
            self.mean_of_features_pooled_train for i in range(len(self.features))
        ]
        self.std_of_features_pooled_train = [
            self.std_of_features_pooled_train for i in range(len(self.features))
        ]
        self.normalize = normalize




class FedHeartDisease(HeartDiseaseRaw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for Heart Disease federated classification.
    One can instantiate this dataset with train or test data coming from either
    of the 4 centers it was created from or all data pooled.
    The train/test split are arbitrarily fixed.

    Parameters
    ----------
    center : int, optional
        Default to 0
    train : bool, optional
        Default to True
    pooled : bool, optional
        Whether to take all data from the 2 centers into one dataset, by
        default False
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    debug : bool, optional,
        Whether or not to use only the part of the dataset downloaded in
        debug mode. Defaults to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    normalize: bool
        Whether or not to normalize the features. We use the corresponding
        training client to compute the mean and std per feature used to
        normalize. When using pooled=True, we use the training part of the full
        dataset to compute the statistics, in order to reflect the differences
        between available informations in FL and pooled mode. Defaults to True.
    """

    def __init__(
        self,
        RawData,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        debug: bool = False,
        data_path: str = None,
        normalize: bool = True,

    ):
        """Cf class description"""




        self.chosen_centers = [RawData.centerr[center]]
        #self.chosen_centers = [center if center in self.centerr else 0]
        print("number of hospitals", len(RawData.centerr))
        print ("Client ID",self.chosen_centers)
        if pooled:
            self.chosen_centers = RawData.centerr

            # We set the apropriate statistics
            self.mean_of_features = RawData.mean_of_features_pooled_train
            self.std_of_features = RawData.std_of_features_pooled_train

        if train:
            self.chosen_sets = ["train"]
        else:
            self.chosen_sets = ["test"]

        to_select = [
            (RawData.sets[idx] in self.chosen_sets)
            and (RawData.centers[idx] in self.chosen_centers)
            for idx, _ in enumerate(RawData.features)
        ]
        print ("whole data")
        print (len(to_select),len(RawData.features))
        self.features = [fp for idx, fp in enumerate(RawData.features) if to_select[idx]]
        self.sets = [fp for idx, fp in enumerate(RawData.sets) if to_select[idx]]
        self.labels = [fp for idx, fp in enumerate(RawData.labels) if to_select[idx]]
        self.centers = [fp for idx, fp in enumerate(RawData.centers) if to_select[idx]]

        print ("This client shape",len(self.features))
        self.mean_of_features = [
            fp for idx, fp in enumerate(RawData.mean_of_features) if to_select[idx]
        ]
        self.std_of_features = [
            fp for idx, fp in enumerate(RawData.std_of_features) if to_select[idx]
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."
        # always normalize
        #if (RawData.normalize)
        X = (self.features[idx] - self.mean_of_features[idx]) / (
                self.std_of_features[idx] + 1e-9)
        #else:
            #X = self.features[idx]

        y = self.labels[idx]
        X = X.reshape((555))
        y = y.reshape((1))
        #print(self.centerr)
        #for idx,_ in enumerate(self.features):
        #    print(self.features[idx])
        return X, y
