import torch.nn as nn
import torch
import numpy as np
from Utils.ValidationUtils import RunningAverage
from Utils.ValidationUtils import MovingAverage
#from DataVisualization import DataVisualization
from Training.EarlyStopping import EarlyStopping
from Utils.ValidationUtils import Metrics
import logging
import Utils.CSVUtils as utils

class ModelTrainer:
    def __init__(self, model, args=None, regime=None):
        self.model = model

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logging.info("[ModelTrainer] " + device)
        self.device = torch.device(device)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.num_epochs = 100

        self.folderPath = "TrainedModels/"

    def GetModel(self):
        return self.model


    def TrainSingleEpoch(self, training_generator):

        self.model.train()
        train_loss_x = MovingAverage()
        train_loss_y = MovingAverage()
        train_loss_z = MovingAverage()
        train_loss_phi = MovingAverage()

        i = 0
        for batch_x, batch_p, batch_r, batch_targets in training_generator:

            batch_targets = batch_targets.to(self.device)
            batch_x = batch_x.to(self.device)
            batch_p = batch_p.to(self.device)
            batch_r = batch_r.to(self.device)
            outputs = self.model(batch_x, batch_p, batch_r)

            loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
            loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
            loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
            loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
            loss = loss_x + loss_y + loss_z + loss_phi

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss_x.update(loss_x)
            train_loss_y.update(loss_y)
            train_loss_z.update(loss_z)
            train_loss_phi.update(loss_phi)

            if (i + 1) % 100 == 0:
                logging.info("[ModelTrainer] Step [{}]: Average train loss {}, {}, {}, {}".format(i+1, train_loss_x.value, train_loss_y.value, train_loss_z.value,
                                                           train_loss_phi.value))
            i += 1

        return train_loss_x.value, train_loss_y.value, train_loss_z.value, train_loss_phi.value


    def ValidateSingleEpoch(self, validation_generator, integer=False):

        self.model.eval()

        valid_loss = RunningAverage()
        valid_loss_x = RunningAverage()
        valid_loss_y = RunningAverage()
        valid_loss_z = RunningAverage()
        valid_loss_phi = RunningAverage()


        y_pred = []
        gt_labels = []
        with torch.no_grad():
            for batch_x, batch_p, batch_r, batch_targets in validation_generator:
                gt_labels.extend(batch_targets.cpu().numpy())
                batch_targets = batch_targets.to(self.device)
                batch_x = batch_x.to(self.device)
                batch_p = batch_p.to(self.device)
                batch_r = batch_r.to(self.device)
                outputs = self.model(batch_x, batch_p, batch_r)

                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi

                valid_loss.update(loss)
                valid_loss_x.update(loss_x)
                valid_loss_y.update(loss_y)
                valid_loss_z.update(loss_z)
                valid_loss_phi.update(loss_phi)

                outputs = torch.stack(outputs, 0)
                outputs = torch.squeeze(outputs)
                outputs = torch.t(outputs)
                y_pred.extend(outputs.cpu().numpy())

        logging.info("[ModelTrainer] Average validation loss {}, {}, {}, {}".format(valid_loss_x.value, valid_loss_y.value,
                                                                  valid_loss_z.value,
                                                                  valid_loss_phi.value))


        return valid_loss_x.value, valid_loss_y.value, valid_loss_z.value, valid_loss_phi.value, y_pred, gt_labels


    def Train(self, training_generator, validation_generator):

        metrics = Metrics()
        early_stopping = EarlyStopping(patience=10, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1),
                                                                    patience=5, verbose=False,
                                                                    threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                                    min_lr=0.1e-6, eps=1e-08)

        for epoch in range(1, self.num_epochs + 1):
            logging.info("[ModelTrainer] Starting Epoch {}".format(epoch))

            train_loss_x, train_loss_y, train_loss_z, train_loss_phi = self.TrainSingleEpoch(training_generator)

            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                validation_generator)

            valid_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi
            scheduler.step(valid_loss)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE, MAE, r2_score = metrics.Update(y_pred, gt_labels,
                                               [train_loss_x, train_loss_y, train_loss_z, train_loss_phi],
                                               [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

            logging.info('[ModelTrainer] Validation MSE: {}'.format(MSE))
            logging.info('[ModelTrainer] Validation MAE: {}'.format(MAE))
            logging.info('[ModelTrainer] Validation r2_score: {}'.format(r2_score))

            checkpoint_filename = self.folderPath + self.model.name + '-{:03d}.pt'.format(epoch)
            early_stopping(valid_loss, self.model, epoch, checkpoint_filename)
            if early_stopping.early_stop:
                logging.info("[ModelTrainer] Early stopping")
                break

        MSEs = metrics.GetMSE()
        MAEs = metrics.GetMAE()
        r2_score = metrics.Get()
        y_pred_viz = metrics.GetPred()
        gt_labels_viz = metrics.GetLabels()
        train_losses_x, train_losses_y, train_losses_z, train_losses_phi, valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi = metrics.GetLosses()

        utils.SaveModelResultsToCSV(MSEs, MAEs, r2_score, gt_labels_viz, y_pred_viz, "Results/train")

        # DataVisualization.desc = "Train_"
        # DataVisualization.PlotLoss(train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi)
        # DataVisualization.PlotMSE(MSEs)
        # DataVisualization.PlotMAE(MAEs)
        # DataVisualization.PlotR2Score(r2_score)
        #
        # DataVisualization.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
        # DataVisualization.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
        # DataVisualization.DisplayPlots()


    def Test(self, test_generator):

        metrics = Metrics()

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            test_generator)

        outputs = y_pred
        outputs = np.reshape(outputs, (-1, 4))
        labels = gt_labels
        y_pred = np.reshape(y_pred, (-1, 4))
        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        MSE, MAE, r2_score = metrics.Update(y_pred, gt_labels,
                                           [0, 0, 0, 0],
                                           [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

        logging.info('[ModelTrainer] Test MSE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MSE[0], MSE[1], MSE[2], MSE[3]))
        logging.info('[ModelTrainer] Test MAE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MAE[0], MAE[1], MAE[2], MAE[3]))
        logging.info('[ModelTrainer] Test r2_score: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(r2_score[0], r2_score[1], r2_score[2], r2_score[3] ))


        return MSE, MAE, r2_score, outputs, labels


