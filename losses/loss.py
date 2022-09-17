import torch.nn as nn
import torch


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.sigmoid = nn.LogSigmoid()

    def forward(self, word_vec, positive_vec, negatives_vecs):

        word_pos_mul = torch.mul(word_vec, positive_vec)
        sum_word_pos_mul = torch.sum(word_pos_mul, 1)
        word_pos_loss = self.sigmoid(sum_word_pos_mul)

        word_neg_mul = torch.bmm(negatives_vecs, word_vec.unsqueeze(-1))
        sum_word_neg_mul = - torch.sum(word_neg_mul, 1)
        word_neg_loss = self.sigmoid(sum_word_neg_mul)
        loss = - (torch.sum(word_pos_loss) + torch.sum(word_neg_loss))

        return loss
