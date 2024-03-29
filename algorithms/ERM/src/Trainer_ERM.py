import wandb

import os
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from algorithms.ERM.src.dataloaders import dataloader_factory
from algorithms.ERM.src.models import model_factory
from torch.utils.data import DataLoader
import torch.nn.functional as F


import torchmetrics
from torchmetrics.classification import MulticlassRecall

from utils.utils import *

class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, z):
        y = self.classifier(z)
        return y

def set_tr_val_samples_labels(meta_filenames, val_size):
    sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels = [], [], [], []

    for idx_domain, meta_filename in enumerate(meta_filenames):
        column_names = ["filename", "class_label"]
        data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\+\s+")
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)

        split_idx = int(len(data_frame) * (1 - val_size))
        sample_tr_paths.append(data_frame["filename"][:split_idx])
        class_tr_labels.append(data_frame["class_label"][:split_idx])

        sample_val_paths.extend(data_frame["filename"][split_idx:])
        class_val_labels.extend(data_frame["class_label"][split_idx:])

    return sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels

def set_test_samples_labels(meta_filenames):
    sample_paths, class_labels = [], []
    for idx_domain, meta_filename in enumerate(meta_filenames):
        column_names = ["filename", "class_label"]
        data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\+\s+")

        sample_paths.extend(data_frame["filename"])
        class_labels.extend(data_frame["class_label"])

    return sample_paths, class_labels

class Trainer_ERM:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.checkpoint_name = (
            "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + exp_idx
        )
        self.plot_dir = (
            "algorithms/" + self.args.algorithm + "/results/plots/" + self.args.exp_name + "_" + exp_idx + "/"
        )

        (
            src_tr_sample_paths,
            src_tr_class_labels,
            src_val_sample_paths,
            src_val_class_labels,
        ) = set_tr_val_samples_labels(self.args.src_train_meta_filenames, self.args.val_size)
        test_sample_paths, test_class_labels = set_test_samples_labels(self.args.target_test_meta_filenames)
        self.train_loaders = []
        for i in range(self.args.n_domain_classes):
            self.train_loaders.append(
                DataLoader(
                    dataloader_factory.get_train_dataloader(self.args.dataset)(
                        src_path=self.args.src_data_path,
                        sample_paths=src_tr_sample_paths[i],
                        class_labels=src_tr_class_labels[i],
                        domain_label=i,
                    ),
                    batch_size=self.args.batch_size,
                    shuffle=True,
                )
            )

        if self.args.val_size != 0:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    src_path=self.args.src_data_path,
                    sample_paths=src_val_sample_paths,
                    class_labels=src_val_class_labels,
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
        else:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    src_path=self.args.src_data_path, sample_paths=test_sample_paths, class_labels=test_class_labels
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )

        self.test_loader = DataLoader(
            dataloader_factory.get_test_dataloader(self.args.dataset)(
                src_path=self.args.src_data_path, sample_paths=test_sample_paths, class_labels=test_class_labels
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(self.args.feature_dim, self.args.n_classes).to(self.device)

        optimizer_params = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(optimizer_params, lr=self.args.learning_rate)

        self.criterion = nn.CrossEntropyLoss()
        self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.args.n_classes).to(self.device)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.args.n_classes).to(self.device)
        self.recall = MulticlassRecall(num_classes=self.args.n_classes).to(self.device)
        self.plotter = UMAPPlot(sample_size=2000)


    def save_plot(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()

        Z_out, Y_out, Y_domain_out = [], [], []
        Z_test, Y_test, Y_domain_test = [], [], []

        with torch.no_grad():
            self.train_iter_loaders = []
            for train_loader in self.train_loaders:
                self.train_iter_loaders.append(iter(train_loader))

            for d_idx in range(len(self.train_iter_loaders)):
                train_loader = self.train_iter_loaders[d_idx]
                for idx in range(len(train_loader)):
                    samples, labels, domain_labels, _ = next(train_loader)
                    samples = samples.to(self.device)
                    labels = labels.to(self.device)
                    domain_labels = domain_labels.to(self.device)
                    z = self.model(samples)
                    Z_out += z.tolist()
                    Y_out += labels.tolist()
                    Y_domain_out += domain_labels.tolist()

            for iteration, (samples, labels, domain_labels, _) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                z = self.model(samples)
                Z_test += z.tolist()
                Y_test += labels.tolist()
                Y_domain_test += domain_labels.tolist()

        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        with open(self.plot_dir + "Z_out.pkl", "wb") as fp:
            pickle.dump(Z_out, fp)
        with open(self.plot_dir + "Y_out.pkl", "wb") as fp:
            pickle.dump(Y_out, fp)
        with open(self.plot_dir + "Y_domain_out.pkl", "wb") as fp:
            pickle.dump(Y_domain_out, fp)

        with open(self.plot_dir + "Z_test.pkl", "wb") as fp:
            pickle.dump(Z_test, fp)
        with open(self.plot_dir + "Y_test.pkl", "wb") as fp:
            pickle.dump(Y_test, fp)
        with open(self.plot_dir + "Y_domain_test.pkl", "wb") as fp:
            pickle.dump(Y_domain_test, fp)

    def train(self):
        self.model.train()
        self.classifier.train()

        n_class_corrected = 0
        total_classification_loss = 0

        total_samples = 0
        slide_names_all = []

        self.train_iter_loaders = []
        for train_loader in self.train_loaders:
            self.train_iter_loaders.append(iter(train_loader))

        for iteration in range(self.args.iterations):
            samples, labels = [], []

            for idx in range(len(self.train_iter_loaders)):
                if (iteration % len(self.train_iter_loaders[idx])) == 0:
                    self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
                train_loader = self.train_iter_loaders[idx]

                itr_samples, itr_labels, itr_domain_labels, sample_paths = next(train_loader)
                if self.args.dataset == 'RCC':
                    slide_names = ['.'.join(sample_path.split('/')[-1].split('_')[-1].split('.')[0:-1]) \
                        for sample_path in sample_paths]
                    slide_names_all.extend(slide_names)
                
                samples.append(itr_samples)
                labels.append(itr_labels)

            samples = torch.cat(samples, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)

            prediction_classes = self.classifier(self.model(samples))
            classification_loss = self.criterion(prediction_classes, labels)
            total_classification_loss += classification_loss.item()

            _, predicted_classes = torch.max(prediction_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)

            self.optimizer.zero_grad()
            classification_loss.backward()
            self.optimizer.step()
            print('iteration:',iteration)
            print('--------')
            print('Accuracy/train',100.0 * n_class_corrected / total_samples)
            print('Loss/train',total_classification_loss / total_samples)
            

            wandb.log({'Accuracy/train': 100.0 * n_class_corrected / total_samples}, step=iteration)
            wandb.log({'Loss/train': total_classification_loss / total_samples}, step=iteration)
            
            if iteration % self.args.step_eval == 0:
                self.evaluate(iteration)

            n_class_corrected = 0
            total_classification_loss = 0
            total_samples = 0

    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()

        n_class_corrected = 0
        total_classification_loss = 0

        latents_all = np.empty((0, self.args.feature_dim))
        labels_all = []
        domain_labels_all = []
        slide_names_all = []
        avg_f1 = 0
        avg_auroc = 0
        avg_recall = 0

        with torch.no_grad():
            # attention: I have changed val_loader to test_loader
            for iteration, (samples, labels, domain_labels, sample_paths) in enumerate(self.test_loader):
                if self.args.dataset == 'RCC':
                    slide_names = ['.'.join(sample_path.split('/')[-1].split('_')[-1].split('.')[0:-1]) \
                        for sample_path in sample_paths]
                    slide_names_all.extend(slide_names)
                samples, labels = samples.to(self.device), labels.to(self.device)
                latent = self.model(samples)
                latents_all = np.vstack((latents_all, latent.detach().cpu().numpy()))
                labels_all.extend(labels.detach().cpu().numpy())
                domain_labels_all.extend(domain_labels.detach().cpu().numpy())
                prediction_classes = self.classifier(latent)
                classification_loss = self.criterion(prediction_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(prediction_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
                avg_auroc += self.auroc(prediction_classes,labels).item()
                avg_f1 += self.f1_score(prediction_classes,labels).item()
                avg_recall += self.recall(prediction_classes,labels).item()
        
        val_acc = 100 * n_class_corrected / len(self.test_loader.dataset)
        val_loss = total_classification_loss / len(self.test_loader.dataset)
        val_auroc = avg_auroc / len(self.test_loader)
        val_f1 = avg_f1 / len(self.test_loader)
        val_recall = avg_recall / len(self.test_loader)

        print('iteration:',iteration)
        print('--------')
        print('Accuracy/test',val_acc, n_iter)
        print('Loss/test',val_loss, n_iter)
        print('AUROC/test', val_auroc)
        print('F1/test', val_f1)
        print('Recall/test', val_recall)
        
        wandb.log({'Accuracy/test': val_acc}, step=n_iter)
        wandb.log({'Loss/test': val_loss}, step=n_iter)
        wandb.log({'AUROC/test': val_auroc}, step=n_iter)
        wandb.log({'F1/test': val_f1}, step=n_iter)
        wandb.log({'Recall/test': val_recall}, step=n_iter)

        plot_latent_class = self.plotter.plot(latents_all, labels_all, slide_names_all, dataset = self.args.dataset, domain = False, \
            name = 'latent_class' + str(n_iter))

        wandb.log({'latent_class': plot_latent_class}, step=n_iter)
        
        self.model.train()
        self.classifier.train()
        if self.args.val_size != 0:
            if self.val_loss_min > val_loss:
                self.val_loss_min = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )
        else:
            if self.val_acc_max < val_acc:
                self.val_acc_max = val_acc
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )

    def test(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()

        n_class_corrected = 0
        latents_all = np.empty((0, self.args.feature_dim))
        probs_all = np.empty((0, self.args.n_classes))
        labels_all = []
        domain_labels_all = []
        slide_names_all = []
        avg_f1 = 0
        avg_auroc = 0
        avg_recall = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels, sample_paths) in enumerate(self.test_loader):
                if self.args.dataset == 'RCC':
                    slide_names = ['.'.join(sample_path.split('/')[-1].split('_')[-1].split('.')[0:-1]) \
                        for sample_path in sample_paths]
                    slide_names_all.extend(slide_names)
                samples, labels = samples.to(self.device), labels.to(self.device)
                latent = self.model(samples)
                latents_all = np.vstack((latents_all, latent.detach().cpu().numpy()))
                labels_all.extend(labels.detach().cpu().numpy())
                domain_labels_all.extend(domain_labels.detach().cpu().numpy())
                prediction_classes = self.classifier(latent)
                probs = F.softmax(prediction_classes)
                probs_all = np.vstack((probs_all, probs.detach().cpu().numpy()))
                _, predicted_classes = torch.max(prediction_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
                avg_auroc += self.auroc(prediction_classes,labels).item()
                avg_f1 += self.f1_score(prediction_classes,labels).item()
                avg_recall += self.recall(prediction_classes,labels).item()

        test_acc = 100 * n_class_corrected / len(self.test_loader.dataset)
        test_auroc = avg_auroc / len(self.test_loader)
        test_f1 = avg_f1 / len(self.test_loader)
        test_recall = avg_recall / len(self.test_loader)

        print('Accuracy/test OOD',test_acc)
        print('AUROC/test OOD', test_auroc)
        print('F1/test OOD', test_f1)
        print('Recall/test OOD', test_recall)

        wandb.log({'Accuracy/test OOD': test_acc})
        wandb.log({'AUROC/test OOD': test_auroc})
        wandb.log({'F1/test OOD': test_f1})
        wandb.log({'Recall/test OOD': test_recall})

        plot_latent_class = self.plotter.plot(latents_all, labels_all, slide_names_all, dataset = self.args.dataset, domain = False, \
            name = 'latent_class OOD')

        wandb.log({'latent_class OOD': plot_latent_class})

        # for WSIs
        if self.args.dataset == 'RCC':
            WSI_probs, WSI_labels = WSI_embedding_aggregator(slide_names_all,\
                                                            probs_all)
            WSI_probs_tensor = torch.from_numpy(WSI_probs).cuda()
            WSI_labels_tensor = torch.from_numpy(WSI_labels).cuda()

            _, WSI_predicted_classes = torch.max(WSI_probs_tensor, 1)

            WSI_accuracy = (WSI_predicted_classes == WSI_labels_tensor).sum()/WSI_labels_tensor.shape[0]
            WSI_auroc = self.auroc(WSI_probs_tensor, WSI_labels_tensor)
            WSI_f1 = self.f1_score(WSI_probs_tensor, WSI_labels_tensor)
            WSI_recall = self.recall(WSI_probs_tensor, WSI_labels_tensor)

            wandb.log({'WSI Accuracy/test OOD': WSI_accuracy})
            wandb.log({'WSI AUROC/test OOD': WSI_auroc})
            wandb.log({'WSI F1/test OOD': WSI_f1})
            wandb.log({'WSI Recall/test OOD': WSI_recall})



