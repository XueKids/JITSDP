import time
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import smote_variants as sv
from sklearn.utils import shuffle


class DataLoader:

    def __init__(self):
        self.base = "../classfication_datasets/"
        self.dataset_names = [
                  'afwall',
                  'alfresco',
                  'AnySoftKeyboard',
                  'apg',
                  'Applozic',
                  'apptentive',
                  'atmosphere',
                  'chat_secure',
                  'deltachat',
                  'facebook',
                  'image',
                  'kiwix_android',
                  'lottie',
                  'openxc',
                  'own_cloud',
                  'PageTurner',
                  'signal',
                  'sync',
                  'syncthing',
                  'wallpaper',
                 ]
        self.dataset = self.__load_data()

    @staticmethod
    def read_xlsx(path):
        df = pd.read_excel(path)
        df.fillna(value=0, inplace=True)   # Handle missing values and change source data
        return df

    def __load_data(self):
        tmp_dataset = OrderedDict()
        for name in self.dataset_names:
            tmp_dataset[name] = self.read_xlsx(self.base + name + '.xlsx')
        return tmp_dataset

    def build_data(self, dataset_name):
        df = self.dataset[dataset_name]
        # stratify split
        train_set, test_set = train_test_split(df, stratify=df['contains_bug'], random_state=42)
        train = train_set.loc[:].values
        x_train = train[:, : -1]
        y_train = train[:, [-1]]
        test = test_set.loc[:].values
        x_test = test[:, : -1]
        y_test = test[:, [-1]]
        y_train = np.squeeze(y_train)  # 将(m ,1)转换为(m,)
        y_test = np.squeeze(y_test)
        oversampler = sv.SYMPROD()
        x_train, y_train = oversampler.sample(x_train, y_train)
        # x_train, y_train = shuffle(x_train, y_train, random_state=42)
        X_train_scaled, X_test_scaled = self.__z_score(x_train, x_test)
        return x_train, x_test, X_train_scaled, X_test_scaled, y_train, y_test

    @staticmethod
    def __z_score(x_train, x_test):
        """
        Standard the features
        """
        scaler_1 = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_trian_scaled = scaler_1.fit_transform(x_train)
        scaler_2 = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_test_scaled = scaler_2.fit_transform(x_test)
        return X_trian_scaled, X_test_scaled