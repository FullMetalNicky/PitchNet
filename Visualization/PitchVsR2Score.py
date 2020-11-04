from __future__ import print_function

from Utils.Logger import *
from Utils.DataProcessor import *
from Training.Dataset import *
from Models.PitchNet import *
from Models.ConvBlock import *
from Training.ModelTrainer import *
from Training.ModelManager import *
from Utils.PerformanceArchiver import *


import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import sys
import sklearn.metrics
import pandas as pd





def OffsetToPitch(h):

    vfov = 65.65
    p_list = np.linspace(0, 160-h, 160-h)
    pitch = -(0.5 * (160 - h) - p_list) * vfov / h

    return pitch


def PlotBasePoint(ax, base_r2_score, mid):
    ax[0][0].scatter(mid, base_r2_score[0],  c='r', label='Base', marker='^')
    ax[0][0].legend(fontsize=15)
    ax[0][1].scatter(mid, base_r2_score[1], c='r', label='Base', marker='^')
    ax[0][1].legend(fontsize=15)
    ax[1][0].scatter(mid, base_r2_score[2], c='r', label='Base', marker='^')
    ax[1][0].legend(fontsize=15)
    ax[1][1].scatter(mid, base_r2_score[3], c='r', label='Base', marker='^')
    ax[1][1].legend(fontsize=15)



def CalculateR2ForPitch(outputs, gt_labels, range_p):

    tot_x_r2 = []
    tot_y_r2 = []
    tot_z_r2 = []
    tot_phi_r2 = []

    for i in range(range_p):
        output = outputs[:, i]
        label = gt_labels[:, i]

        x = output[:, 0]
        x_gt = label[:, 0]
        y = output[:, 1]
        y_gt = label[:, 1]
        z = output[:, 2]
        z_gt = label[:, 2]
        phi = output[:, 3]
        phi_gt = label[:, 3]

        x_r2 = sklearn.metrics.r2_score(x_gt, x)
        y_r2 = sklearn.metrics.r2_score(y_gt, y)
        z_r2 = sklearn.metrics.r2_score(z_gt, z)
        phi_r2 = sklearn.metrics.r2_score(phi_gt, phi)
        tot_x_r2.append(x_r2)
        tot_y_r2.append(y_r2)
        tot_z_r2.append(z_r2)
        tot_phi_r2.append(phi_r2)


    return tot_x_r2, tot_y_r2, tot_z_r2, tot_phi_r2


def PlotModelR2Score(ax, y_values, x_values, color, title, model_label):
    ax.plot(x_values, y_values, color=color, label=model_label)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Pitch', fontsize=18)
    ax.set_ylabel('R2', fontsize=18)
    ax.set_ylim([0, 1])
    ax.set_ymargin(0.2)
    plt.legend(fontsize=15)




def VizPitchvsR2ScoreSubPlots(ax, outputs, gt_labels, p_test, color, model_label):
    min_p = np.min(p_test)
    max_p = np.max(p_test)

    range_p = max_p - min_p + 1
    outputs = np.reshape(outputs, (-1, range_p, 4))
    gt_labels = np.reshape(gt_labels, (-1, range_p, 4))
    tot_pitch = OffsetToPitch(96)

    tot_x_r2, tot_y_r2, tot_z_r2, tot_phi_r2 = CalculateR2ForPitch(outputs, gt_labels, range_p)

    PlotModelR2Score(ax[0][0], tot_x_r2, tot_pitch, color, "Output variable: x", model_label)
    PlotModelR2Score(ax[0][1], tot_y_r2, tot_pitch, color, "Output variable: y", model_label)
    PlotModelR2Score(ax[1][0], tot_z_r2, tot_pitch, color, "Output variable: z", model_label)
    PlotModelR2Score(ax[1][1], tot_phi_r2, tot_pitch, color, "Output variable: phi", model_label)

    return range_p


def Plot2Models(p_test, archive_name1, archive_name2, base_r2_score1):


    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("R2 Score as a function of Pitch", fontsize=22)


    outputs, gt_labels = LoadPerformanceResults(archive_name1)
    range_p = VizPitchvsR2ScoreSubPlots(ax, outputs, gt_labels, p_test, 'b', 'Pitch-input')

    outputs, gt_labels = LoadPerformanceResults(archive_name2)
    range_p = VizPitchvsR2ScoreSubPlots(ax, outputs, gt_labels, p_test, 'g', 'Pitch-augmented')

    PlotBasePoint(ax, base_r2_score1, 0)

    plt.subplots_adjust(hspace=0.3)


    # if name.find(".pickle"):
    #     name = name.replace(".pickle", '')
    name = "modelcomparison"
    plt.savefig(name + '_pitch.png')
    plt.show()


def Plot1Model(p_test, archive_name, base_r2_score):

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("R2 Score as a function of Pitch")

    outputs, gt_labels = LoadPerformanceResults(archive_name)

    range_p = VizPitchvsR2ScoreSubPlots(ax, outputs, gt_labels, p_test, 'b', 'new')


    PlotBasePoint(ax, base_r2_score, 0)

    if archive_name.find(".pickle"):
        archive_name = archive_name.replace(".pickle", '')
    plt.savefig(archive_name + '_pitch.png')
    plt.show()



def main():
    archive_name = "../Scripts/Archive/PitchNetOthersCrop64.pickle"
    model_name = "PitchNet.pt"
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    datset_path = DATA_PATH + "160x96HimaxTest16_4_2020Cropped64.pickle"
    p_test = DataProcessor.GetPitchFromTestData(datset_path)
    base_r2_score = [0.8628, 0.7919, 0.7068, 0.4467]


    #Plot1Model(p_test, archive_name, model_name, base_r2_score)

    archive_name2 = "../Scripts/Archive/Others160x96HimaxTest16_4_2020Cropped64.pickle"
    Plot2Models(p_test, archive_name, archive_name2, base_r2_score)


if __name__ == '__main__':
    main()