import torch
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from torchvision.transforms import functional as FV
import torchvision.transforms as T
import random



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
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        
        # self.loss_func = losses.TripletMarginLoss(margin)
        self.loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(), 
				     reducer = ThresholdReducer(high=0.3), 
			 	     embedding_regularizer = LpRegularizer())

    def forward(self, inputs, targets):
        hard_pairs = self.miner(inputs, targets)
        loss = self.loss_func(inputs, targets, hard_pairs)
        return loss


class AngularTripletLoss(nn.Module):

    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self):
        super(AngularTripletLoss, self).__init__()
        # self.miner = miners.AngularMiner(angle=20)
        
        # self.loss_func = losses.TripletMarginLoss(margin)
        self.loss_func = losses.AngularLoss(alpha=40, 
				     reducer = ThresholdReducer(high=0.3), 
			 	     embedding_regularizer = LpRegularizer())

    def forward(self, inputs, targets):
        # hard_pairs = self.miner(inputs, targets)
        loss = self.loss_func(inputs, targets)
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


class to_uint8_tensor(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, sample):
        normalized_tensor = (sample-sample.min())/(sample.max()-sample.min())
        normalized_tensor = normalized_tensor * 255.0  # case [0, 1]
        normalized_tensor = torch.clip(normalized_tensor, 0.0, 255.0)
        return FV.to_pil_image(normalized_tensor.type(torch.uint8))

from torch.utils.data import Dataset

class SSLBatchDataloader(Dataset):
    def __init__(self, batch_data):
        self.self_supervision_transform = T.Compose([
                    T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                    T.ToTensor(),
                    # T.RandomErasing(),
                    # T.RandomGrayscale(p = 0.35)
                    ]) 
        self.batch_data = batch_data

    def get_image(self, idx):
        sample = self.batch_data[idx,:,:,:]
        normalized_tensor = (sample-sample.min())/(sample.max()-sample.min())
        normalized_tensor = normalized_tensor * 255.0  # case [0, 1]
        normalized_tensor = torch.clip(normalized_tensor, 0.0, 255.0)
        return self.self_supervision_transform(FV.to_pil_image(normalized_tensor.type(torch.uint8)))

    def __len__(self):
        return self.batch_data.shape[0]

    def __getitem__(self, index):
        sample = self.get_image(index)
        return sample