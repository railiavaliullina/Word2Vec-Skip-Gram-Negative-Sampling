import numpy as np
import os
import time
import torch
from torch.optim.lr_scheduler import StepLR

from datasets.Text8Dataset import Text8Dataset
from models.Word2Vec import get_model
from utils.logging import Logger
from losses.loss import Loss


class Trainer(object):
    def __init__(self, train_cfg, dataset_cfg, model_cfg):
        """
        Class for initializing and performing training procedure.
        :param train_cfg: train config
        :param dataset_cfg: dataset config
        :param model_cfg: model config
        """
        self.train_cfg = train_cfg
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg

        self.dataset = Text8Dataset(dataset_cfg)
        self.model = get_model(model_cfg, dataset_cfg)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.logger = Logger(self.train_cfg)

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_cfg.lr, momentum=self.train_cfg.momentum,
                                    nesterov=self.train_cfg.nesterov, weight_decay=self.train_cfg.weight_decay)
        return optimizer

    @staticmethod
    def get_criterion():
        """
        Returns Loss class instance as criterion.
        :return: criterion
        """
        criterion = Loss()
        return criterion

    def restore_model(self):
        """
        Restores saved model.
        """
        if self.train_cfg.load_saved_model:
            print(f'Trying to load checkpoint from epoch {self.train_cfg.epoch_to_load}...')
            try:
                checkpoint = torch.load(self.train_cfg.checkpoints_dir +
                                        f'/checkpoint_{self.train_cfg.epoch_to_load}.pth')
                load_state_dict = checkpoint['model']
                self.model.load_state_dict(load_state_dict)
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                self.optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint from epoch {self.train_cfg.epoch_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')

    def save_model(self):
        """
        Saves model.
        """
        if self.train_cfg.save_model and self.epoch % self.train_cfg.epochs_saving_freq == 0:
            print('Saving current model...')
            state = {
                'model': self.model.state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': self.optimizer.state_dict(),
            }
            if not os.path.exists(self.train_cfg.checkpoints_dir):
                os.makedirs(self.train_cfg.checkpoints_dir)

            path_to_save = os.path.join(self.train_cfg.checkpoints_dir,
                                        f'checkpoint_{self.epoch}_{self.global_step}.pth')
            torch.save(state, path_to_save)
            print(f'Saved model to {path_to_save}.')

    def make_training_step(self, batch):
        """
        Makes single training step.
        :param batch: current batch containing input vector and it`s label
        :return: loss on current batch
        """
        target_tensor, context_tensor, negative_tensor = batch
        word_vec, positive_vec, negatives_vecs = self.model(target_tensor, context_tensor, negative_tensor)
        loss = self.criterion(word_vec, positive_vec, negatives_vecs)

        assert not torch.isnan(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        """
        Runs training procedure.
        """
        total_training_start_time = time.time()
        self.start_epoch, self.epoch, self.global_step = 0, -1, 0

        # restore model if necessary
        self.restore_model()

        # start training
        print(f'Starting training...')

        dl = self.dataset.get_custom_dataloader()
        iter_num = len(dl)
        print(f'Iterations num: {iter_num}')

        scheduler = StepLR(self.optimizer,
                           step_size=iter_num // ((self.train_cfg.lr/self.train_cfg.fin_lr) * self.train_cfg.lr_gamma),
                           gamma=self.train_cfg.lr_gamma)

        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            epoch_start_time = time.time()
            self.epoch = epoch
            print(f'Epoch: {self.epoch}/{self.train_cfg.epochs}')

            losses = []

            for iter_, batch in enumerate(dl):
                loss = self.make_training_step(batch)
                losses.append(loss)

                self.logger.log_metrics(names=['train/loss'], metrics=[loss], step=self.global_step)

                if iter_ % 1e2:
                    print(
                        f'iter: {iter_}/{iter_num}, step: {self.global_step}, epoch: {self.epoch}, '
                        f'loss: {np.mean(losses[-50:])}, lr: {self.optimizer.param_groups[0]["lr"]}')

                if iter_ % 1e5 == 0 and iter_ != 0:
                    self.save_model()

                self.global_step += 1
                scheduler.step()

            self.epoch += 1
            self.save_model()
            self.logger.log_metrics(names=['train/mean_loss_per_epoch'], metrics=[np.mean(losses)], step=self.epoch)

            print(f'Epoch total time: {round((time.time() - epoch_start_time) / 60, 3)} min')
        print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')
