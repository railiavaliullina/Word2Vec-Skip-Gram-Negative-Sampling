from easydict import EasyDict


cfg = EasyDict()

cfg.lr = 0.025
cfg.fin_lr = 0.0001
cfg.lr_gamma = 0.1
cfg.weight_decay = 1e-4
cfg.momentum = 0.99
cfg.nesterov = True
cfg.epochs = 1

cfg.log_metrics = False
cfg.experiment_name = 'word2vec_last'

cfg.load_saved_model = False
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 0
cfg.save_model = True
cfg.epochs_saving_freq = 1
