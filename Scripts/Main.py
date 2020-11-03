
import numpy as np
import cv2
import pandas as pd
from torchsummary import summary
import torch

from Utils.Logger import *
from Utils.DataProcessor import *
from Training.Dataset import *
from Models.PitchNet import *
from Models.ConvBlock import *
from Training.ModelTrainer import *

DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"



def main():
    LogProgram()

    train_path = DATA_PATH + "160x96HimaxTrain16_4_2020AugCrop.pickle"
    [x_train, x_validation, p_train, p_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(train_path)

    training_set = Dataset(x_train, p_train, y_train, True)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 0}
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_validation, p_validation, y_validation)
    validation_generator = data.DataLoader(validation_set, **params)

    model = PitchNet(ConvBlock, [1, 1, 1], True)
    trainer = ModelTrainer(model)
    trainer.Train(training_generator, validation_generator)

if __name__ == '__main__':
    main()