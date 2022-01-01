"""
MIT License (SOURCE: https://github.com/jma712/DIRECT)

Copyright (c) 2021 Jing Ma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# change to the latest version 2020.10.25

import math
import torch
from torch import nn
from time import time
from ncore.apps.util import info
from torch.nn import functional as F


class MTVAE(nn.Module):
    def __init__(self, num_train_samples: int,
                 num_treatments: int, mu_p_wt: float = 1.0,
                 device: torch.device = torch.device("cpu"), k: int = 4,
                 nogb: bool = False, dim_zt: int = 32, dim_zi: int = 32,
                 beta: float = 20.0):
        super(MTVAE, self).__init__()

        self.beta = beta
        self.mu_p_wt = mu_p_wt
        self.k = k
        self.device = device

        dim_input_t = num_train_samples
        dim_input_i = num_treatments
        self.num_cluster = k
        self.nogb = nogb
        self.dim_zt = dim_zt
        self.dim_zi = dim_zi

        # Recognition model
        # q(Z^T|A)
        self.logvar_zt = nn.Sequential(nn.Linear(dim_input_t, dim_zt))  #nn.Tanh()
        self.mu_zt = nn.Sequential(nn.Linear(dim_input_t, dim_zt))
        # q(C|Z^T)
        self.qc = nn.Sequential(nn.Linear(dim_zt, dim_zt), nn.ReLU(), nn.Linear(dim_zt, k))
        # p(a|z^T)
        self.a_reconstby_zt = nn.Sequential(nn.Linear(self.dim_zt, dim_input_t))
        # q(Z^(I,K)|A,C)
        self.mu_zi_k = nn.Linear(dim_input_i, self.dim_zi)
        self.logvar_zi_k = nn.Linear(dim_input_i, self.dim_zi)

        # prior generator
        # p(Z^T|C)
        self.mu_p_zt = torch.normal(mean=0, std=1, size=(self.k, self.dim_zt), device=self.device)
        self.mu_p_zt = torch.nn.Parameter(self.mu_p_zt, requires_grad=True)
        self.register_parameter("mu_p_zt", self.mu_p_zt)
        self.mu_p_wt = self.mu_p_wt
        self.logvar_p_zt = torch.nn.Parameter(torch.ones((self.k, self.dim_zt),device=self.device), requires_grad=True)
        self.register_parameter("logvar_p_zt", self.logvar_p_zt)

        # Generative model
        # predict outcome
        self.y_pred_1 = nn.Linear(self.dim_zt, self.num_cluster * self.dim_zi)
        self.y_pred_2 = nn.Linear(self.num_cluster * self.dim_zi, 1)
        self.logvar_y = nn.Linear(self.dim_zt, self.num_cluster * self.dim_zi)

    def loss_function(self, input_ins_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list,
                      zi_sample_list, a_pred, mu_y, logvar_y, target, a_reconstby_zt, input_treat_trn):
        # 1. recontrust loss
        loss_bce = nn.BCELoss(reduction='mean').to(self.device)
        loss_reconst = loss_bce(a_pred.view(-1), input_ins_batch.reshape(-1))

        # 2. KLD_C
        KLD_C = torch.mean(torch.sum(qc * torch.log(self.k * qc + 1e-10), dim=1), dim=0)

        # 3. E_KLD_QT_PT
        mu_zt = mu_zt.unsqueeze(-1)
        logvar_zt = logvar_zt.unsqueeze(-1)

        mu_p_zt = mu_p_zt.T
        logvar_p_zt = logvar_p_zt.T
        mu_p_zt = mu_p_zt.unsqueeze(0)
        logvar_p_zt = logvar_p_zt.unsqueeze(0)

        KLD_QT_PT = 0.5 * (((logvar_p_zt - logvar_zt) + (
                    (logvar_zt.exp() + (mu_zt - self.mu_p_wt * mu_p_zt).pow(2)) / logvar_p_zt.exp())) - 1)

        # zt
        loss_bce2 = nn.BCELoss(reduction='mean').to(self.device)
        loss_reconst_zt = loss_bce2(a_reconstby_zt.reshape(-1), input_treat_trn.reshape(-1))

        qc = qc.unsqueeze(-1)  # m x k x 1
        qc = qc.expand(-1, self.k, 1)  # m x k x 1

        E_KLD_QT_PT = torch.mean(torch.sum(torch.bmm(KLD_QT_PT, qc), dim=1), dim=0)

        mu_zi_all = None
        log_zi_all = None
        for k in range(self.k):
            mu_zi_k = mu_zi_list[k]
            logvar_zi_k = logvar_zi_list[k]
            mu_zi_all = mu_zi_k if mu_zi_all is None else torch.cat([mu_zi_all, mu_zi_k], dim=1)
            log_zi_all = logvar_zi_k if log_zi_all is None else torch.cat([log_zi_all, logvar_zi_k], dim=1)
        KL_ZI = -0.5 * torch.sum(1 + log_zi_all - mu_zi_all.pow(2) - log_zi_all.exp(), dim=1)  # n
        KL_ZI = torch.mean(KL_ZI, dim=0)

        # 5. loss_y
        temp = 0.5 * math.log(2 * math.pi)
        target = target.view(-1, 1)

        bb = - 0.5 * ((target - mu_y).pow(2)) / logvar_y.exp() - 0.5 * logvar_y - temp
        loss_y = - torch.mean(
            torch.sum(- 0.5 * ((target - mu_y).pow(2)) / logvar_y.exp() - 0.5 * logvar_y - temp, dim=1), dim=0)

        # MSE_Y
        loss_mse = nn.MSELoss(reduction='mean')
        loss_y_mse = loss_mse(mu_y, target)

        # 6. loss balance
        loss_balance = 0.0

        loss = loss_reconst + KL_ZI + KLD_C + E_KLD_QT_PT + loss_y

        eval_result = {
            'loss': loss, 'loss_reconst': loss_reconst, 'KLD_C': KLD_C, 'E_KLD_QT_PT': E_KLD_QT_PT,
            'loss_reconst_zt': loss_reconst_zt,
            'KL_ZI': KL_ZI, 'loss_y': loss_y, 'loss_y_mse': loss_y_mse, 'loss_balance': loss_balance,
        }

        return eval_result

    def encode_t(self, input_treat):
        mu_zt = self.mu_zt(input_treat)
        logvar_zt = self.logvar_zt(input_treat)

        return mu_zt, logvar_zt

    def get_treat_rep(self, input_treat_new):
        # encoder: zt, zi
        mu_zt, logvar_zt = self.encode_t(input_treat_new)
        zt_sample = self.reparameterize(mu_zt, logvar_zt)  # m x d
        qc = self.compute_qc(zt_sample)  # m x k, unnormalized logits
        cates = F.softmax(qc, dim=1)  # normalize with softmax, m x k

        return mu_zt, logvar_zt, cates

    def compute_qc(self, zt_sample, type='cos'):
        # zt_sample: m x d_t
        if type == 'cos':
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            zt_sample_K = torch.unsqueeze(zt_sample, dim=0).repeat(self.num_cluster, 1, 1)  # K x m x d_t
            mu_p_zt = torch.unsqueeze(self.mu_p_zt, dim=1).repeat(1, zt_sample_K.shape[1], 1)
            cos_similarity = cos(self.mu_p_wt * mu_p_zt, zt_sample_K)
            qc = cos_similarity.T
            qc *= 10
        elif type == 'linear':
            qc = self.qc(zt_sample)
        elif type == 'euc':

            zt_sample_K = torch.unsqueeze(zt_sample, dim=0).repeat(self.num_cluster, 1, 1)  # K x m x d_t
            mu_p_zt = torch.unsqueeze(self.mu_p_zt, dim=1).repeat(1, zt_sample_K.shape[1], 1)

            distance = torch.norm(zt_sample_K - self.mu_p_wt * mu_p_zt, dim=-1)  # k x m
            qc = 1.0/(distance.T + 1)
        return qc

    def encode_i(self, input_ins, qc):
        mu_zi_list = []
        logvar_zi_list = []

        # c sample
        if self.training:
            cates = nn.functional.gumbel_softmax(logits=qc, hard=True)  # one-hot, m x k
        else:
            cates = F.softmax(qc, dim=1)  # normalize with softmax, m x k

        # q(z^(I,k)|a^k)
        for k in range(self.num_cluster):
            cates_k = cates[:, k].reshape(1, -1)  # 1 x m, cates_k[j]=1: item j is in cluster k

            # q-network
            x_k = input_ins * cates_k
            mu_k = self.mu_zi_k(x_k)
            logvar_k = self.logvar_zi_k(x_k)

            mu_zi_list.append(mu_k)
            logvar_zi_list.append(logvar_k)

        return mu_zi_list, logvar_zi_list, cates

    def reparameterize(self, mu, logvar):
        '''
        z = mu + std * epsilon
        '''
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # p(A|Z^T,Z^I,C)
    def decoder(self, zt_sample, zi_sample_list, c_sample):
        probs = None
        for k in range(self.num_cluster):
            cates_k = c_sample[:, k].reshape(1, -1)

            zi_sample_k = zi_sample_list[k]  # n x d
            a_pred_k = torch.matmul(zi_sample_k, zt_sample.T)  # n x m
            a_pred_k = torch.sigmoid(a_pred_k)
            a_pred_k = a_pred_k * cates_k
            probs = (a_pred_k if (probs is None) else (probs + a_pred_k))

        return probs

    def predictY(self, zt_sample, zi_sample_list, c_sample, adj_batch):
        # concat zi
        zi_all = None  # batch x (K x d)
        for k in range(len(zi_sample_list)):  # every cluster
            Z_i_k = zi_sample_list[k]
            zi_all = Z_i_k if zi_all is None else torch.cat([zi_all, Z_i_k], dim=1)

        a_zt = torch.matmul(adj_batch, zt_sample)  # batch size x d_t

        rep_w1 = self.y_pred_1(a_zt)
        pred_y2 = self.y_pred_2(zi_all)
        mu_y = torch.matmul(rep_w1, zi_all.T).diag().view(-1, 1) + pred_y2

        logvar_y = torch.ones_like(mu_y).to(self.device)

        return mu_y, logvar_y

    def forward(self, input_ins, input_treat):
        # encoder: zt, zi
        mu_zt, logvar_zt = self.encode_t(input_treat)
        zt_sample = self.reparameterize(mu_zt, logvar_zt)  # sample zt: m x d
        qc = self.compute_qc(zt_sample)  # m x k, unnormalized logits
        qc = F.softmax(qc, dim=1)  # normalize with softmax, m x k

        cates = qc

        mu_zi_list, logvar_zi_list, c_sample = self.encode_i(input_ins, qc)

        a_reconstby_zt = self.a_reconstby_zt(zt_sample)
        a_reconstby_zt = torch.sigmoid(a_reconstby_zt)  # m x n

        # sample zi
        zi_sample_list = []  # size = k, each elem is n x d
        for k in range(self.num_cluster):
            mu_zi_k = mu_zi_list[k]
            logvar_zi_k = logvar_zi_list[k]
            zi_sample_k = self.reparameterize(mu_zi_k, logvar_zi_k)
            zi_sample_list.append(zi_sample_k)

        # decoder
        a_pred = self.decoder(zt_sample, zi_sample_list, c_sample)
        mu_y, logvar_y = self.predictY(zt_sample, zi_sample_list, c_sample, input_ins)  # n x 1

        return mu_zt, logvar_zt, self.mu_p_zt, self.logvar_p_zt, cates, mu_zi_list, logvar_zi_list, zi_sample_list, \
               a_pred, mu_y, logvar_y, a_reconstby_zt

    def fit(self, epochs, num_treatments, trn_loader, input_treat_trn, optimizer, active_opt=None):
        if active_opt is None:
            active_opt = [True, True, True, True]
        self.train()

        dim_input_t = num_treatments
        optimizer_1 = optimizer[0]
        optimizer_2 = optimizer[1]
        optimizer_3 = optimizer[2]
        optimizer_4 = optimizer[3]

        for epoch in range(epochs):
            start_time = time()

            loss_1, loss_2, loss_3, loss_4, num_batches_seen = 0, 0, 0, 0, 0
            for batch_idx, batch_data in enumerate(trn_loader):
                _, adj_batch, target = \
                    batch_data[:, :-1-dim_input_t], \
                    batch_data[:, -1-dim_input_t:-1].reshape((-1, dim_input_t)),\
                    batch_data[:, -1]
                num_batches_seen += 1

                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                optimizer_3.zero_grad()
                optimizer_4.zero_grad()

                # forward pass
                if active_opt[0]:
                    for i in range(5):
                        optimizer_1.zero_grad()

                        mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, \
                            a_pred, mu_y, logvar_y, a_reconstby_zt = self(adj_batch, input_treat_trn)

                        eval_result = self.loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt,
                                                         qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y,
                                                         logvar_y, target, a_reconstby_zt, input_treat_trn)
                        loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                            eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result[
                                'loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result[
                                'KLD_C'], eval_result['loss_y'], eval_result['loss_y_mse']
                        # backward propagation
                        loss_1 += loss_a_reconst_zt
                        loss_a_reconst_zt.backward()
                        optimizer_1.step()

                if active_opt[2]:
                    for i in range(3):
                        optimizer_3.zero_grad()

                        mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, \
                            a_pred, mu_y, logvar_y, a_reconstby_zt = self(adj_batch, input_treat_trn)

                        eval_result = self.loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt,
                                                         qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred,
                                                         mu_y, logvar_y, target, a_reconstby_zt, input_treat_trn)
                        loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                            eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result[
                                'loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result[
                                'KLD_C'], \
                            eval_result['loss_y'], eval_result['loss_y_mse']

                        # backward propagation
                        pm_beta = 1.0 if epoch < 100 else self.beta
                        beta_loss = loss_reconst + pm_beta * KL_ZI
                        loss_2 += beta_loss
                        beta_loss.backward()
                        optimizer_3.step()

                if active_opt[3]:
                    for i in range(20):
                        optimizer_4.zero_grad()

                        mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, \
                            a_pred, mu_y, logvar_y, a_reconstby_zt = self(adj_batch, input_treat_trn)

                        eval_result = self.loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt,
                                                         qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y,
                                                         logvar_y, target, a_reconstby_zt, input_treat_trn)
                        loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                            eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result[
                                'loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result[
                                'KLD_C'], \
                            eval_result['loss_y'], eval_result['loss_y_mse']

                        # backward propagation
                        loss_3 += loss_y
                        loss_y.backward()
                        optimizer_4.step()

                # optimize for the centroid
                if active_opt[1]:
                    for i in range(20):
                        optimizer_2.zero_grad()

                        # forward pass
                        mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, \
                            a_pred, mu_y, logvar_y, a_reconstby_zt = self(adj_batch, input_treat_trn)

                        eval_result = self.loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc,
                                                         mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y,
                                                         logvar_y, target, a_reconstby_zt, input_treat_trn)
                        loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                            eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result[
                                'loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result[
                                'KLD_C'], \
                            eval_result['loss_y'], eval_result['loss_y_mse']

                        # backward propagation
                        kl_loss = 5 * KLD_C + E_KLD_QT_PT
                        loss_4 += kl_loss
                        kl_loss.backward()
                        optimizer_2.step()

            epoch_duration = time() - start_time
            info("Epoch {:d}/{:d} [{:.2f}s]: recon. loss = {:.4f}, zi. loss = {:.4f}, "
                 "y loss = {:.4f}, cent. loss = {:.4f}"
                 .format(epoch, epochs, epoch_duration,
                         float((loss_1/num_batches_seen).detach().numpy()),
                         float((loss_2/num_batches_seen).detach().numpy()),
                         float((loss_3/num_batches_seen).detach().numpy()),
                         float((loss_4/num_batches_seen).detach().numpy())))
