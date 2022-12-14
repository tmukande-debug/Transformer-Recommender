#!/usr/bin/env python36
# -*- coding: utf-8 -*-

######################################################
# Adapted from CRIPAC-DIG/SR-GNN for fair comparison #
######################################################

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
# from torch.nn import TransformerEncoder
# from torch.nn import TransformerEncoderLayer
from nfnets.agc import AGC
#from conformer import ConformerEncoder
#from x_transformers import TransformerWrapper, Encoder
import torch
#from flash_cosine_sim_attention import flash_cosine_sim_attention
from mega_pytorch import Mega




class SelfAttentionNetwork(Module):
    def __init__(self, opt, n_node):
        super(SelfAttentionNetwork, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        # self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        #self.transformerEncoderLayer = TransformerEncoderLayer(dim=self.hidden_size, num_tokens = 256,  depth = 6, ema_heads=opt.nhead, dim_feedforward=self.hidden_size * opt.feedforward)
        # self.transformerEncoder = TransformerEncoder(dim=self.hidden_size, ema_heads=opt.nhead, dim_feedforward=self.hidden_size * opt.feedforward)
        self.transformerEncoder = Mega(dim=self.hidden_size, ema_heads=opt.nhead, dim_feedforward=self.hidden_size * opt.feedforward)
        # print(self.n_node)
        
            
        # self.final_linear = nn.Linear(64, 1)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.AGC_optim = AGC(self.parameters(), self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # def compute_scores(self, hidden, mask):
    #     ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
    #     b = self.embedding.weight[1:]  # n_nodes x latent_size
    #     scores = torch.matmul(ht, b.transpose(1, 0))
    #     return scores

    # def forward(self, inputs, A):
    #     hidden = self.embedding(inputs)
    #     hidden = hidden.transpose(0,1).contiguous()
    #     hidden = self.transformerEncoder(hidden)
    #     hidden = hidden.transpose(0,1).contiguous()
    #     return hidden

    def forward(self, inputs):
        # print(inputs.size())
        hidden = self.transformerEncoder(inputs)
        # hidden = hidden.transpose(0,1).contiguous()
        # print(hidden.size())
        hidden = torch.matmul(hidden, self.transformerEncoder.item_embed.weight[1:].transpose(1, 0)) # weight tying
        # hidden = self.final_linear(hidden)
        # hidden = torch.squeeze(hidden, -1)
        # print(hidden.size())

        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
