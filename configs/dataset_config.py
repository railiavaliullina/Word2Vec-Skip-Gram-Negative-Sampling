from easydict import EasyDict


cfg = EasyDict()


cfg.batch_size = 64
cfg.vocab_size = 50000
cfg.context_win_size = 2
cfg.negatives_num = 20

cfg.ds_path = '../data/text8/text8'
cfg.ds_path_test = '../data/text8/questions-words.txt'
cfg.load_data_samples = False
cfg.training_samples_path = '../data/training_samples.pickle'
