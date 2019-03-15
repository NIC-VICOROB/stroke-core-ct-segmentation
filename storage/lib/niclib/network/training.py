import copy
import os
import sys
import string
from abc import ABC, abstractmethod
import time
import datetime
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

#
from niclib.network.loss_functions import *
from niclib.network.optimizers import NICOptimizer
from niclib.inout.train_log import TrainingLogger

from niclib.network.layers import *

class NIC_Trainer:
    def __init__(self, loss_func, optimizer, batch_size, max_epochs=100, train_metrics=None, eval_metrics=None,
                 print_interval=0.4, device=torch.device('cuda'), do_train=True):
        assert isinstance(optimizer, NICOptimizer)

        self.do_train = do_train

        # Training config
        self.device = device
        self.optimizer = optimizer
        self.train_loss_func = loss_func
        self.bs = batch_size
        self.max_epochs = max_epochs

        # Metrics
        self.train_metric_funcs = {'loss': copy.copy(loss_func)}
        if train_metrics is not None:
            self.train_metric_funcs.update(train_metrics)

        self.test_metric_funcs = {'loss': copy.copy(loss_func)}
        if eval_metrics is not None:
            self.test_metric_funcs.update(eval_metrics)

        # Printing options
        self.print_interval = print_interval

        # Execution variables
        self.current_epoch = -1
        self.print_flag = True
        self.print_lock = threading.Lock()
        self.print_timer = None

    def train(self, model, train_gen, val_gen):
        print("Training model for {} epochs".format(self.max_epochs))
        model = model.to(self.device)
        model_optimizer = self.optimizer.set_parameters(model.parameters())

        self.current_epoch = -1
        for epoch_num in range(self.max_epochs):
            self.current_epoch += 1
            print("Epoch {}/{}".format(self.current_epoch, self.max_epochs))

            # Train current epoch and test on validation set
            epoch_train_metrics = self.train_epoch(model, model_optimizer, train_gen)
            epoch_test_metrics = self.test_epoch(model, val_gen)

        print("Training finished\n")

    def train_epoch(self, model, optimizer, train_gen):
        model.train()

        train_metrics = dict()
        for k in self.train_metric_funcs.keys():
            train_metrics['train_{}'.format(k)] = torch.tensor(0.0).to(self.device)

        self.print_timer = threading.Timer(self.print_interval, self._setPrintFlag)
        self.print_timer.start()

        eta = ElapsedTimeEstimator(len(train_gen))
        for batch_idx, (x, y) in enumerate(train_gen):
            optimizer.zero_grad()  # Reset accumulated gradients

            # Send generated x,y batch to GPU
            if isinstance(x, list):
                for i in range(len(x)):
                    x[i] = x[i].to(self.device)
            else:
                x = x.to(self.device)

            if isinstance(y, list):
                for i in range(len(y)):
                    y[i] = y[i].to(self.device)
            else:
                y = y.to(self.device)

            # Forward pass and loss computation
            y_pred = model(x)
            loss = self.train_loss_func(y_pred, y)

            loss.backward()  # Auto gradient loss
            optimizer.step()  # Backpropagate the loss

            # Update training metrics
            for k, eval_func in self.train_metric_funcs.items():
                train_metrics['train_{}'.format(k)] += eval_func(y_pred, y).item()

            # PRINTING LOGIC
            self.print_lock.acquire()
            if self.print_flag:
                # Compute average metrics
                avg_metrics = dict(train_metrics)
                for k, v in avg_metrics.items():
                    avg_metrics[k] = float(v.item() / (batch_idx + 1))

                self._printProgressBar(batch_idx, len(train_gen), eta.update(batch_idx + 1), avg_metrics)

                self.print_flag, self.print_timer = False, threading.Timer(self.print_interval, self._setPrintFlag)
                self.print_timer.start()
            self.print_lock.release()
        self.print_timer.cancel()

        avg_metrics = dict(train_metrics)
        for k, v in avg_metrics.items():
            avg_metrics[k] = float(v.item() / len(train_gen))

        self._printProgressBar(len(train_gen), len(train_gen), eta.get_elapsed_time(), avg_metrics)
        return avg_metrics

    def test_epoch(self, model, val_gen):
        model.eval()

        test_metrics = dict()
        for k in self.test_metric_funcs.keys():
            test_metrics['val_{}'.format(k)] = 0.0

        with torch.no_grad():  # Turns off autograd (faster exec)
            for batch_idx, (x, y) in enumerate(val_gen):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x)

                for k, eval_func in self.test_metric_funcs.items():
                    test_metrics['val_{}'.format(k)] += eval_func(y_pred, y)

        # Compute average metrics
        for k, v in test_metrics.items():
            test_metrics[k] = float(v / len(val_gen))

        # Print average validation metrics at the end of the training bar
        for k, v in test_metrics.items():
            print(' - {}={:0<6.4f}'.format(k, v), end='')

        return test_metrics

    def _setPrintFlag(self, value=True):
        self.print_lock.acquire()
        self.print_flag = value
        self.print_lock.release()

    def _printProgressBar(self, batch_num, total_batches, eta, metrics):
        length, fill = 25, '='
        percent = "{0:.1f}".format(100 * (batch_num / float(total_batches)))
        filledLength = int(length * batch_num // total_batches)
        bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

        metrics_string = ' - '.join(['{}={:0<6.4f}'.format(k, v) for k, v in metrics.items()])

        print('\r [{}] {}/{} ({}%) ETA {} - {}'.format(
            bar, batch_num, total_batches, percent, eta, metrics_string), end='')
        sys.stdout.flush()



class EarlyStoppingTrain:
    def __init__(self, loss_func, optimizer, batch_size, max_epochs=100, train_metrics=None, eval_metrics=None, early_stopping_metric='val_loss',
                 early_stopping_patience=1, print_interval=0.4, device=torch.device('cuda'), do_train=True):
        assert isinstance(optimizer, NICOptimizer)

        self.do_train = do_train

        # Training config
        self.device = device
        self.optimizer = optimizer
        self.train_loss_func = loss_func
        self.bs = batch_size
        self.max_epochs = max_epochs
        self.train_logger = TrainingLogger()

        # Metrics
        self.train_metric_funcs = {'loss': copy.copy(loss_func)}
        if train_metrics is not None:
            self.train_metric_funcs.update(train_metrics)

        self.test_metric_funcs = {'loss': copy.copy(loss_func)}
        if eval_metrics is not None:
            self.test_metric_funcs.update(eval_metrics)


        # Early stopping config
        self.early_stopping_metric = 'val_loss' if early_stopping_metric is None else early_stopping_metric
        self.patience = early_stopping_patience
        self.min_delta = 1E-4

        # Printing options
        self.print_interval = print_interval

        # Execution variables
        self.current_epoch = -1
        self.print_flag = True
        self.print_lock = threading.Lock()
        self.print_timer = None

        assert self.early_stopping_metric in self.test_metric_funcs.keys()

    def train(self, model, train_gen, val_gen, checkpoint_filepath, log_filepath):
        print("Training model for {} epochs".format(self.max_epochs))
        model = model.to(self.device)
        model_optimizer = self.optimizer.set_parameters(model.parameters())

        if self.do_train:
            self.current_epoch = -1
            best_metric = dict(epoch=-1, value=sys.float_info.max)
            for epoch_num in range(self.max_epochs):
                self.current_epoch += 1
                print("Epoch {}/{}".format(self.current_epoch, self.max_epochs))

                # Train current epoch and test on validation set
                epoch_train_metrics = self.train_epoch(model, model_optimizer, train_gen)
                epoch_test_metrics = self.test_epoch(model, val_gen)

                # Log epoch results
                if log_filepath is not None:
                    now = datetime.datetime.now()
                    basic_params = {'Epoch': self.current_epoch, 'Date': now.strftime("%Y-%m-%d"), 'Time':now.strftime('%H:%M:%S')}
                    self.train_logger.add_epoch_params({**basic_params, **epoch_train_metrics, **epoch_test_metrics})
                    self.train_logger.write_to_csv(log_filepath)

                monitored_metric_value = epoch_test_metrics['val_{}'.format(self.early_stopping_metric)]
                if best_metric['value'] - monitored_metric_value > self.min_delta:
                    print(' (best)', sep='')
                    best_metric.update(epoch=self.current_epoch, value=monitored_metric_value)
                    torch.save(model, checkpoint_filepath)
                else:
                    print('')

                if self.current_epoch - best_metric['epoch'] >= self.patience:
                    break
        else:
            print("Skipped training ('do train' is disabled)")
            torch.save(model, checkpoint_filepath)

        print("Training finished\n")

    def train_epoch(self, model, optimizer, train_gen):
        model.train()

        train_metrics = dict()
        for k in self.train_metric_funcs.keys():
            train_metrics['train_{}'.format(k)] = torch.tensor(0.0).to(self.device)

        self.print_timer = threading.Timer(self.print_interval, self._setPrintFlag)
        self.print_timer.start()

        eta = ElapsedTimeEstimator(len(train_gen))
        for batch_idx, (x, y) in enumerate(train_gen):
            optimizer.zero_grad()  # Reset accumulated gradients

            # Send generated x,y batch to GPU
            if isinstance(x, list):
                for i in range(len(x)):
                    x[i] = x[i].to(self.device)
            else:
                x = x.to(self.device)

            if isinstance(y, list):
                for i in range(len(y)):
                    y[i] = y[i].to(self.device)
            else:
                y = y.to(self.device)

            # Forward pass and loss computation
            y_pred = model(x)
            loss = self.train_loss_func(y_pred, y)

            loss.backward()  # Auto gradient loss
            optimizer.step()  # Backpropagate the loss

            # try:
            #     from niclib.architecture.Ensemble_WAVG import draw_wavg_weights
            #     model.apply(draw_wavg_weights)
            # except Exception:
            #     print("Couldn't draw stuff")

            # Update training metrics
            for k, eval_func in self.train_metric_funcs.items():
                train_metrics['train_{}'.format(k)] += eval_func(y_pred, y).item()

            # PRINTING LOGIC
            self.print_lock.acquire()
            if self.print_flag:
                # Compute average metrics
                avg_metrics = dict(train_metrics)
                for k, v in avg_metrics.items():
                    avg_metrics[k] = float(v.item() / (batch_idx + 1))

                self._printProgressBar(batch_idx, len(train_gen), eta.update(batch_idx + 1), avg_metrics)

                self.print_flag, self.print_timer = False, threading.Timer(self.print_interval, self._setPrintFlag)
                self.print_timer.start()
            self.print_lock.release()
        self.print_timer.cancel()

        avg_metrics = dict(train_metrics)
        for k, v in avg_metrics.items():
            avg_metrics[k] = float(v.item() / len(train_gen))

        self._printProgressBar(len(train_gen), len(train_gen), eta.get_elapsed_time(), avg_metrics)
        return avg_metrics

    def test_epoch(self, model, val_gen):
        model.eval()

        test_metrics = dict()
        for k in self.test_metric_funcs.keys():
            test_metrics['val_{}'.format(k)] = 0.0

        with torch.no_grad():  # Turns off autograd (faster exec)
            for batch_idx, (x, y) in enumerate(val_gen):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x)

                for k, eval_func in self.test_metric_funcs.items():
                    test_metrics['val_{}'.format(k)] += eval_func(y_pred, y)

        # Compute average metrics
        for k, v in test_metrics.items():
            test_metrics[k] = float(v / len(val_gen))

        # Print average validation metrics at the end of the training bar
        for k, v in test_metrics.items():
            indicator = '*' if 'val_{}'.format(self.early_stopping_metric) is k else ''
            print(' - {}{}={:0<6.4f}'.format(indicator, k, v), end='')

        return test_metrics

    def _setPrintFlag(self, value=True):
        self.print_lock.acquire()
        self.print_flag = value
        self.print_lock.release()

    def _printProgressBar(self, batch_num, total_batches, eta, metrics):
        length, fill = 25, '='
        percent = "{0:.1f}".format(100 * (batch_num / float(total_batches)))
        filledLength = int(length * batch_num // total_batches)
        bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

        metrics_string = ' - '.join(['{}={:0<6.4f}'.format(k, v) for k,v in metrics.items()])

        print('\r [{}] {}/{} ({}%) ETA {} - {}'.format(
            bar, batch_num, total_batches, percent, eta, metrics_string), end='')
        sys.stdout.flush()


class ElapsedTimeEstimator:
    def __init__(self, total_iters, update_weight=0.05):
        self.total_eta = None
        self.start_time = time.time()
        self.total_iters = total_iters

        self.last_iter = {'num': 0, 'time': time.time()}
        self.update_weight = update_weight

    def update(self, current_iter_num):
        current_eta, current_iter = None, {'num': current_iter_num, 'time':time.time()}
        if current_iter['num'] > self.last_iter['num']:
            iters_between = current_iter['num'] - self.last_iter['num']
            time_between = current_iter['time'] - self.last_iter['time']
            current_eta = (time_between / iters_between) * (self.total_iters - current_iter['num'])
            if self.total_eta is None:
                self.total_eta = current_eta

            w = self.update_weight
            self.total_eta = (current_eta * w) + ((self.total_eta - time_between) * (1 - w))

        self.last_iter = current_iter
        # Return formatted eta in hours, minutes, seconds
        return self._format_time_interval(self.total_eta) if current_eta is not None else '?'

    def get_elapsed_time(self):
        return self._format_time_interval(time.time() - self.start_time)

    @staticmethod
    def _format_time_interval(seconds):
        time_format = "%M:%S"
        if seconds > 3600:
            time_format = "%H:%M:%S"
            if seconds > 24 * 3600:
                time_format = "%d days, %H:%M:%S"

        formatted_time = time.strftime(time_format, time.gmtime(seconds))
        return formatted_time