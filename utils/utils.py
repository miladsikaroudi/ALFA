import torch
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from torchvision.transforms import functional as FV
from torch.utils.data import Dataset
import torchvision.transforms as T
from . import common_functions as c_f
from . import losses_and_miners_utils as lmu


import torch
from torch import nn

class TripletMarginLoss(nn.Module):

    """ Sampling Matters in Deep Embedding Learning: https://arxiv.org/pdf/1706.07567.pdf
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()

        self.miner = miners.TripletMarginMiner(margin=1.5, type_of_triplets="semihard")

        # self.miner = miners.MultiSimilarityMiner()

        # self.loss_func = losses.MarginLoss(margin=3.0, 
        #         nu=0, 
        #         beta=1.2, 
        #         # triplets_per_anchor="all", 
        #         # learn_beta=False, 
        #         # num_classes=None, 
        #         reducer = ThresholdReducer(low=0.3),
        #         # embedding_regularizer = LpRegularizer()
        #         )

        self.loss_func = TripletMarginLossCollapsePrevention(margin, 
                # nu=0, 
                # beta=1.2, 
                # # triplets_per_anchor="all", 
                # # learn_beta=False, 
                # # num_classes=None, 
                # reducer = ThresholdReducer(low=0.3),
                # # embedding_regularizer = LpRegularizer()
                )

    def forward(self, inputs, targets):
        semihard_pairs = self.miner(inputs, targets)
        loss = self.loss_func(inputs, targets, semihard_pairs)
        return loss

class TripletMarginLossCollapsePrevention(losses.BaseMetricLossFunction):

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)
        current_margins /= (an_dists.mean() + 1e-17)
        violation = current_margins + self.margin
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.nn.functional.relu(violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

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

class SSLBatchDataloader(Dataset):
    def __init__(self, batch_data):
        self.self_supervision_transform = T.Compose([
                    # T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
