import torch
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses



import torch
from torch import nn

class TripletLoss(nn.Module):

    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        self.loss_func = losses.TripletMarginLoss(margin)

    def forward(self, inputs, targets):
        hard_pairs = self.miner(inputs, targets)
        loss = self.loss_func(inputs, targets, hard_pairs)
        return loss


def kd(data1=None, label1=None, data2=None, label2=None, bool_indicator=None, n_class=3, temperature=2.0):
    label1 = F.one_hot(label1 , n_class).double()
    label2 = F.one_hot(label2 , n_class).double()
    kd_loss = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []
    bool_indicator = torch.tensor([[1.0], [1.0], [1.0]]).cuda()


    for cls in range(n_class):
        mask1 = torch.tile(torch.unsqueeze(label1[:, cls], -1), [1, n_class])
        logits_sum1 = torch.mean(torch.multiply(data1, mask1), axis=0)
        num1 = torch.mean(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps)# add eps for prevent un-sampled class resulting in NAN
        prob1 = torch.nn.functional.softmax(activations1 / temperature)
        prob1 = torch.clip(prob1, min=1e-8, max=1.0)  # for preventing prob=0 resulting in NAN

        mask2 = torch.tile(torch.unsqueeze(label2[:, cls], -1), [1, n_class])
        logits_sum2 = torch.mean(torch.multiply(data2, mask2), axis=0)
        num2 = torch.mean(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        prob2 = torch.nn.functional.softmax(activations2 / temperature)
        prob2 = torch.clip(prob2, min=1e-8, max=1.0)

        KL_div = (torch.mean(prob1 * torch.log(prob1 / prob2)) + torch.mean(prob2 * torch.log(prob2 / prob1))) / 2.0
        kd_loss += KL_div * bool_indicator[cls]
        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = kd_loss / n_class

    return kd_loss, prob1s, prob2s