import torch
import numpy as np
from collections import Counter
import random
import pickle
from torch.autograd import Variable


class Text8Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        """
        Class for reading, preprocessing, encoding and sampling data.
        :param cfg: dataset config
        """
        self.cfg = cfg

        self.read_data()
        self.get_vocab()
        self.run_subsampling()
        self.get_negative_sampling_probs()

        if not self.cfg.load_data_samples:
            self.get_training_samples()
        else:
            with open(self.cfg.training_samples_path, 'rb') as f:
                self.training_samples = pickle.load(f)

    def read_data(self):
        """
        Reads source data.
        """
        print('Reading data for train...')
        with open(self.cfg.ds_path, 'r') as f:
            self.text = f.read().split()
        self.words_num = len(self.text)

        counts = dict(Counter(self.text))
        norm_f = np.sum(np.asarray(list(counts.values())))
        self.counts_dict = {w: counts[w] / float(norm_f) for w in counts}

    def get_vocab(self):
        self.vocabulary = np.asarray(sorted(Counter(Counter(self.text)).items(),
                                            key=lambda item: item[1],
                                            reverse=True))[:self.cfg.vocab_size, 0]

        self.word_to_index, self.index_to_word = {}, {}
        for w, word in enumerate(self.vocabulary):
            self.word_to_index[word] = w
            self.index_to_word[w] = word

    def run_subsampling(self):
        print(f'Subsampling...')
        self.subsampled_text = []

        for w, word in enumerate(self.text):
            keep_prob = random.random()
            word_weight = (np.sqrt(self.counts_dict[word] * 1e3) + 1) * 1e-3 / float(self.counts_dict[word])
            if word_weight > keep_prob:
                self.subsampled_text.append(word)

            if w % 1e6 == 0:
                print(f'subsampling, word: {w}/{self.words_num}')

        self.subsampled_text_size = len(self.subsampled_text)
        self.subsampled_words_dict = dict(Counter(self.subsampled_text))
        self.subsampled_words_counts = list(self.subsampled_words_dict.values())

    def get_negative_sampling_probs(self):
        negative_sampling_probs = {}
        norm_f = np.sum([count ** 0.75 for count in self.subsampled_words_counts])
        for word in self.subsampled_text:
            negative_sampling_probs[word] = self.subsampled_words_dict[word] ** 0.75 / norm_f

        self.sampling_probs = np.asarray(list(negative_sampling_probs.values()))
        self.words_sample_from = np.asarray(list(negative_sampling_probs.keys()))

    def get_negatives(self, sample_size):
        while True:
            negatives = []
            chosen_negatives = np.random.choice(self.words_sample_from, size=sample_size,
                                                replace=False, p=self.sampling_probs)
            for neg_word in chosen_negatives:
                while self.word_to_index.get(neg_word, None) is None:
                    neg_word = np.random.choice(self.words_sample_from, size=1, p=self.sampling_probs)[0]
                negatives.append(neg_word)
            yield negatives

    def get_training_samples(self):

        print('Getting training samples...')
        training_samples = []
        negatives = self.get_negatives(self.cfg.negatives_num)

        for w, word in enumerate(self.subsampled_text):
            if self.word_to_index.get(word, None) is not None:

                left_idx = np.max([0, w - self.cfg.context_win_size])
                right_idx = np.min([w + self.cfg.context_win_size + 1, self.subsampled_text_size])
                unk_words_num = int(np.sum([1 for w in self.subsampled_text[left_idx:right_idx]
                                            if self.word_to_index.get(w, None) is None]))
                right_idx += unk_words_num

                for idx in range(left_idx, right_idx):
                    if self.word_to_index.get(self.subsampled_text[idx], None) is not None and idx != w:
                        training_samples.append((word, self.subsampled_text[idx], next(negatives)))

            if w % 1000 == 0:
                print(f'got training sample for word: {w}/{self.subsampled_text_size}')

        print(f'Number of training samples: {len(training_samples)}')
        self.training_samples = training_samples
        self.training_samples_num = len(self.training_samples)

        with open('../data/training_samples.pickle', 'wb') as f:
            pickle.dump(training_samples, f)
        print('Saved training_samples in ../data/training_samples.pickle')

    def get_custom_dataloader(self):
        word_, positive_, negatives_, samples = [], [], [], []
        for idx, (word, positive, negatives) in enumerate(self.training_samples):

            word_idx, positive_idx = self.word_to_index[word], self.word_to_index[positive]
            negatives_idx = [self.word_to_index[w] for w in negatives]

            word_.append(word_idx)
            positive_.append(positive_idx)
            negatives_.append(negatives_idx)

            if (idx + 1) % self.cfg.batch_size == 0 or idx == self.training_samples_num - 1:
                samples.append((Variable(torch.tensor(word_)),
                                Variable(torch.tensor(positive_)),
                                Variable(torch.tensor(negatives_))))
                word_, positive_, negatives_ = [], [], []

        return samples
