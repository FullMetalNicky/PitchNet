import pandas as pd
import numpy as np
import random
import logging
import cv2
import sys

class DataProcessor:

    @staticmethod
    def GetSizeDataFromDataFrame(dataset):

        h = int(dataset['h'].values[0])
        w = int(dataset['w'].values[0])
        c = int(dataset['c'].values[0])

        return h, w, c

    @staticmethod
    def CreateSizeDataFrame(h, w, c):

        sizes_df = pd.DataFrame({'c': c, 'w': w, 'h': h}, index=[0])

        return sizes_df

    @staticmethod
    def ProcessTrainData(trainPath):
        """Reads the .pickle file and converts it into a format suitable fot training

            Parameters
            ----------
            trainPath : str
                The file location of the .pickl

            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """
        train_set = pd.read_pickle(trainPath)

        logging.info('[DataProcessor] train shape: ' + str(train_set.shape))
        size = train_set.shape[0]
        n_val = int(float(size) * 0.2)
        #n_val = 13000

        h, w, c = DataProcessor.GetSizeDataFromDataFrame(train_set)

        np.random.seed(1749)
        random.seed(1749)
        # split between train and test sets:
        x_train = train_set['x'].values
        x_train = np.vstack(x_train[:]).astype(np.float32)
        x_train = np.reshape(x_train, (-1, h, w, c))

        x_train= np.swapaxes(x_train, 1, 3)
        x_train = np.swapaxes(x_train, 2, 3)

        y_train = train_set['y'].values
        y_train = np.vstack(y_train[:]).astype(np.float32)

        p_train = train_set['p'].values
        p_train = np.vstack(p_train[:]).astype(np.float32)
        vfov = 65.65
        p_train = -(0.5*(160 - h) - p_train) * vfov / h

        r_train = np.zeros((size, 1)).astype(np.float32)


        ix_val, ix_tr = np.split(np.random.permutation(train_set.shape[0]), [n_val])
        x_validation = x_train[ix_val, :]
        x_train = x_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]
        p_validation = p_train[ix_val, :]
        p_train = p_train[ix_tr, :]
        r_validation = r_train[ix_val, :]
        r_train = r_train[ix_tr, :]

        shape_ = len(x_train)

        sel_idx = random.sample(range(0, shape_), k=(size-n_val))
        x_train = x_train[sel_idx, :]
        y_train = y_train[sel_idx, :]
        p_train = p_train[sel_idx, :]
        r_train = r_train[sel_idx, :]

        return [x_train, x_validation, p_train, p_validation,  r_train, r_validation, y_train, y_validation]

    @staticmethod
    def ProcessTestData(testPath, isExtended=False):
        """Reads the .pickle file and converts it into a format suitable fot testing

            Parameters
            ----------
            testPath : str
                The file location of the .pickle
            isExtended : bool, optional
                True if the dataset contains both head and hand pose and you wish to retrieve both


            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """

        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        h, w, c = DataProcessor.GetSizeDataFromDataFrame(test_set)

        x_test = test_set['x'].values
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, h, w, c))


        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = test_set['y'].values
        y_test = np.vstack(y_test[:]).astype(np.float32)

        if isExtended ==True:
            z_test = test_set['z'].values
            z_test = np.vstack(z_test[:]).astype(np.float32)
            return [x_test, y_test, z_test]


        return [x_test, y_test]


    @staticmethod
    def GetPitchFromTestData(testPath):

        """Reads the .pickle file and extracts the pitch values

                   Parameters
                   ----------
                   testPath : str
                       The file location of the .pickle

                   Returns
                   -------
                   list
                       list of pitch  values
                   """

        p_test = None
        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        if 'p' in test_set.columns:
            p_test = test_set['p'].values

        return p_test





