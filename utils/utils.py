import torch
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from .augmentation_utils import *

from sklearn.decomposition import PCA
import umap

from . import common_functions as c_f
from . import losses_and_miners_utils as lmu

from collections import defaultdict, OrderedDict
import numpy as np
from numpy import mean
from operator import itemgetter 

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Serif'
rcParams['font.size'] = 24

############################### defined loss utils ##############################
class TripletMarginLoss(nn.Module):

    """ Sampling Matters in Deep Embedding Learning: https://arxiv.org/pdf/1706.07567.pdf
    """
    def __init__(self, margin_loss, margin_semihard):
        super(TripletMarginLoss, self).__init__()

        self.miner = miners.TripletMarginMiner(margin=margin_semihard, type_of_triplets="semihard")

        self.loss_func = TripletMarginLossCollapsePrevention(margin=margin_loss, 
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
        current_margins /= abs(an_dists.mean() - ap_dists.mean() + 1e-3)
        violation = current_margins + self.margin
        if self.smooth_loss:
            loss = F.softplus(violation)
        else:
            loss = F.relu(violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

class KLDiv(object):
    def __init__(self) -> None:
        self.temperature = 2.0
        self.eps = 1e-16

    def __call__(self, data1=None, label1=None, data2=None, label2=None, bool_indicator=None, n_class= None):
        label1 = F.one_hot(label1 , n_class).double()
        label2 = F.one_hot(label2 , n_class).double()
        kd_loss = 0.0
        prob1s = []
        prob2s = []
        bool_indicator = torch.ones(n_class,1).cuda()

        for cls in range(n_class):
            mask1 = torch.tile(torch.unsqueeze(label1[:, cls], -1), [1, n_class])
            logits_sum1 = torch.mean(torch.multiply(data1, mask1), axis=0)
            num1 = torch.mean(label1[:, cls])
            activations1 = logits_sum1 * 1.0 / (num1 + self.eps)# add eps for prevent un-sampled class resulting in NAN
            prob1 = F.softmax(activations1 / self.temperature)
            prob1 = torch.clip(prob1, min=1e-8, max=1.0)  # for preventing prob=0 resulting in NAN

            mask2 = torch.tile(torch.unsqueeze(label2[:, cls], -1), [1, n_class])
            logits_sum2 = torch.mean(torch.multiply(data2, mask2), axis=0)
            num2 = torch.mean(label2[:, cls])
            activations2 = logits_sum2 * 1.0 / (num2 + self.eps)
            prob2 = F.softmax(activations2 / self.temperature)
            prob2 = torch.clip(prob2, min=1e-8, max=1.0)

            prob3 = F.one_hot(torch.tensor(cls) , n_class).double().cuda()
            prob3 = (0.9-(0.1/(n_class-1)))*prob3 + 0.1/(n_class-1)

            KL_div = (torch.mean(prob1 * torch.log(prob1 / prob2)) +  torch.mean(prob1 * torch.log(prob1 / prob3))+\
                        torch.mean(prob2 * torch.log(prob2 / prob1)) + torch.mean(prob2 * torch.log(prob2 / prob3))+\
                            torch.mean(prob3 * torch.log(prob3 / prob1)) + torch.mean(prob3 * torch.log(prob3 / prob2)))/ 6.0
            kd_loss += KL_div * bool_indicator[cls]
            prob1s.append(prob1)
            prob2s.append(prob2)

        kd_loss = kd_loss / n_class

        return kd_loss, prob1s, prob2s



############################### list utils ##############################
class to_uint8_tensor(object):
    def __call__(self, sample):
        normalized_tensor = (sample-sample.min())/(sample.max()-sample.min())
        normalized_tensor = normalized_tensor * 255.0  # case [0, 1]
        normalized_tensor = torch.clip(normalized_tensor, 0.0, 255.0)
        return FV.to_pil_image(normalized_tensor.type(torch.uint8))

def pylist_unique(py_list: list)->list:
    unique_list = []
    if type(type(py_list) == type(unique_list)):
        # traverse for all elements
        for elem in py_list:
            # check if exists in unique_list or not
            if elem not in unique_list:
                unique_list.append(elem)
    return unique_list

def merge_path_embedding_list(list1 : list, list2: np.ndarray)-> list:
    merged_list = []
    if list1 != [] and list2 != [] and len(list1) == len(list2):
        merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list
    
def list_to_dict_average(init_list):
    dictionary = defaultdict(list)
    for k, v in init_list:
        dictionary[k].append(v)
    return {key: tuple(map(mean, zip(*value))) for key, value in dictionary.items()}

############################### visualizarion utils ##############################
class UMAPPlot(object):
    def __init__(self, sample_size:int) -> None:
        np.random.seed(0)
        self.sample_size = sample_size
        self.pca = PCA(n_components=50)
        self.umap_reducer = umap.UMAP()

    def plot(self, embeddings, labels, WSI_names, dataset:str, domain: bool, name:str):

            n_classes = len(pylist_unique(labels))
            idx = np.random.choice(embeddings.shape[0], min(embeddings.shape[0],self.sample_size), replace=False)

            sampled_embedding = embeddings[ idx,:]
            sampled_labels = list(itemgetter(*idx)(labels))
            
            embedding_embedded_50d = self.pca.fit_transform(sampled_embedding)
            embedding_embedded = self.umap_reducer.fit_transform(embedding_embedded_50d)

            fig, _ = plt.subplots(1, figsize=(14, 10))
            if dataset == 'RCC':
                sampled_WSIs = list(itemgetter(*idx)(WSI_names))
                class_list_visualization = classnames_factory(dataset='RCC')
                class_names = [class_list_visualization[str(i)] for i in range(len(class_list_visualization))]
                newcmp = shrink_cmap(cmap_official = 'RdPu')
                WSI_embeddings, WSI_labels = WSI_embedding_aggregator(sampled_WSIs, embedding_embedded)
                plt.scatter(embedding_embedded[:, 0], embedding_embedded[:, 1], s=450, c=sampled_labels, cmap=newcmp, alpha=0.15, linewidths=0.01, edgecolors='b')
                plt.scatter(WSI_embeddings[:, 0], WSI_embeddings[:, 1], s=650, c=WSI_labels, cmap=newcmp, alpha=1.0, edgecolors='black')
            elif dataset == 'PACS' or dataset =='synthetic':
                class_list_visualization = classnames_factory(dataset=dataset)
                class_names = [class_list_visualization[str(i)] for i in range(len(class_list_visualization))]
                plt.scatter(embedding_embedded[:, 0], embedding_embedded[:, 1], s=80, c=sampled_labels, cmap='jet', alpha=1.0, linewidths=0.1, edgecolors='b')
            cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
            cbar.set_ticks(np.arange(n_classes))
            if not domain:
                cbar.set_ticklabels(class_names)
            plt.axis('off')
            plt.title(name)
            return fig

def WSI_embedding_aggregator(WSI_name_list, embedding_list):
    filenames_embedding_list = merge_path_embedding_list(WSI_name_list, embedding_list)
    dict_of_avg_tuples = list_to_dict_average(filenames_embedding_list)
    ordered_dict_of_avg_tuples = OrderedDict(dict_of_avg_tuples.copy())
    df_of_slide_classes = pd.read_csv('file_name_class_index_dictionary.csv')
    slide_classes_set = {(k,int(df_of_slide_classes[df_of_slide_classes['file_name']==k]['cases.0.diagnoses.0.primary_diagnosis'])) for k in ordered_dict_of_avg_tuples.keys()}
    slide_classes_dict = dict(slide_classes_set)
    classidx_embedding_set = {(v, slide_classes_dict[k]) for k, v in ordered_dict_of_avg_tuples.items()}
    classidx_embedding_dict = OrderedDict(classidx_embedding_set)
    WSI_embeddings = np.asarray(list(classidx_embedding_dict.keys()))
    WSI_labels = np.asarray(list(classidx_embedding_dict.values()))
    return WSI_embeddings, WSI_labels

def shrink_cmap(cmap_official='RdPu'):
    rdpu = cm.get_cmap(cmap_official, 256)
    newcolors = rdpu(np.linspace(0, 1, 256))
    blue = np.array([255/256, 205/256, 1/256, 1])
    newcolors[:25, :] = blue
    newcmp = ListedColormap(newcolors)
    return newcmp

def classnames_factory(dataset):
    if dataset == 'RCC':
        class_list_visualization = {'0': 'ccRCC',
                                    '1': 'pRCC',
                                    '2': 'crRCC'}
    elif dataset == 'PACS':
        class_list_visualization = {'0': 'dog',
                                    '1': 'elephant',
                                    '2': 'giraffe',
                                    '3': 'guitar',
                                    '4': 'horse',
                                    '5': 'house',
                                    '6': 'person'}
    elif dataset == 'synthetic':
        class_list_visualization = {'0': 'HP',
                                    '1': 'SSA'}               
    return class_list_visualization