import torch.nn as nn


class Word2Vec(nn.Module):

    def __init__(self, model_cfg, dataset_cfg):
        """
        Class for building Word2Vec model.
        :param model_cfg: model config
        :param dataset_cfg: dataset config
        """
        super(Word2Vec, self).__init__()

        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg

        self.w_input = nn.Embedding(self.dataset_cfg.vocab_size, self.model_cfg.hidden_layer_size)
        if self.model_cfg.init_w_input_with_xavier:
            nn.init.xavier_uniform_(self.w_input.weight)

        self.w_output = nn.Embedding(self.dataset_cfg.vocab_size, self.model_cfg.hidden_layer_size)
        if self.model_cfg.init_w_output_with_zeros:
            nn.init.constant_(self.w_output.weight, 0)

    def forward(self, word, positive, negatives):
        word_vec = self.w_input(word)
        positive_vec = self.w_output(positive)
        negatives_vecs = self.w_output(negatives)
        return word_vec, positive_vec, negatives_vecs


def get_model(model_cfg, dataset_cfg):
    """
    Gets MLP model.
    :return: MLP model
    """
    model = Word2Vec(model_cfg, dataset_cfg)
    return model
