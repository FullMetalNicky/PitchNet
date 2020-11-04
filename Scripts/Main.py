
import numpy as np
import cv2
import pandas as pd
#from torchsummary import summary
import torch

from Utils.Logger import *
from Utils.DataProcessor import *
from Training.Dataset import *
from Models.PitchNet import *
from Models.ConvBlock import *
from Training.ModelTrainer import *
from Training.ModelManager import *
from Utils.PerformanceArchiver import *

DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"


def train():
    train_path = DATA_PATH + "160x96HimaxTrain16_4_2020AugCrop.pickle"
    [x_train, x_validation, p_train, p_validation, r_train, r_validation, y_train,
     y_validation] = DataProcessor.ProcessTrainData(train_path)

    training_set = Dataset(x_train, p_train, r_train, y_train, True)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 0}
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_validation, p_validation, r_validation, y_validation)
    validation_generator = data.DataLoader(validation_set, **params)

    model = PitchNet(ConvBlock, [1, 1, 1], True)
    trainer = ModelTrainer(model)
    trainer.Train(training_generator, validation_generator)


def test():
    model = PitchNet(ConvBlock, [1, 1, 1], True)
    ModelManager.Read("PitchNet.pt", model)

    trainer = ModelTrainer(model)

    test_path = DATA_PATH + "160x96HimaxTest16_4_2020.pickle"
    [x_test, p_test, r_test, y_test] = DataProcessor.ProcessTestData(test_path)
    test_set = Dataset(x_test, p_test, r_test, y_test)

    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}
    test_loader = data.DataLoader(test_set, **params)
    MSE, MAE, r2_score, outputs, labels = trainer.Test(test_loader)



def archive():

    datset_path = DATA_PATH + "160x96HimaxTest16_4_2020Cropped64.pickle"
    model_path = "PitchNet.pt"
    name = "Archive/PitchNetOthersCrop64.pickle"

    DumpPerformanceResults(datset_path, model_path, name)

def main():
    LogProgram()

    #train()
    #test()
    archive()



if __name__ == '__main__':
    main()