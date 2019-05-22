#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-13 22:58:26
# @Author  : skyhiter

import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.attn = nn.Sequential(nn.Tanh(), nn.Linear(layer_size, 1, bias=False), nn.Softmax(dim=1))

    def forward(self, batch_input):
        # batch_input shape (batch, max_len, embd_dim)
        # 归一化的attn weight
        attn_w = self.attn(batch_input)  #attn_w shape (batch,max_len,1)
        # attentive representation
        attn_r = torch.matmul(batch_input.transpose(dim0=1, dim1=2), attn_w)
        attn_w = attn_w.squeeze(2)
        attn_r = attn_r.squeeze(2)
        # attn_r shape (batch, embd_dim), attn_w shape (batch, max_len)
        return attn_r, attn_w


class LSTMClassifier(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim, num_layers=1, dropout=0.5, bidirectional=True):
        super().__init__()
        self.vocab_size = embeddings.size(0)
        self.embedding_dim = embeddings.size(1)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
            padding_idx=1)  # torchtext中的vocab.vectors表默认: 0是unk，1是pad
        # self.embedding.weight是Parameter对象,它是Tensor的子类
        # self.embedding.weight.data是Tensor对象
        self.embedding.weight.data.copy_(embeddings)  # 加载预训练词向量
        self.embedding.weight.requires_grad = True  # finetuning词向量
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional)
        # 双向
        self.output = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.output_dim)

        self.attn = AttentionLayer(self.hidden_dim)
        self.attn_bi = AttentionLayer(self.hidden_dim * 2)

    def forward(self, batch):
        sentences, lengths = batch  # torchtext默认不是batch first
        # 假设 sentences是4 * 3的（4是句子长度，3是batch）， lengths是[4,3 2]，sentences已按长度降序排列
        # 假设这是一个3层的双向LSTM，mebedding 是 20 * 11（即一共20个vocab，每个词向量的长度是11）, hidden_dim 是 5，output_dim 是2
        # sentences 就是一个batch，已经按照batch内降序排列
        # sentences shape: (length, batch)
        embeded = self.embedding(sentences)  # 注意embedding表应该是 "每行是一个词的向量", sentences是4 * 3，则embeded是4 * 3 * 11的
        # embeded_packed 是一个 PackedSequence 对象,做了压缩（扁平化处理）
        # embeded_packed 样子是 (tensor shape[9, 11], batch_sizes=[3, 3, 2, 1])，这里的batch_size并不是SGD中的batch
        # sentences 是 4 * 3（显然是pad后的了），而embeded_packed[0]是[9, 11]，也就是说把4*3拼接成了1列，删除了pad（就是说一共9个有效词）, 11是embedding_dim
        # 这里的batch_size是指，把一个batch按句子长度从大到小排列，横着看(因为不是batch first)的有效词个数（除pad后）
        embeded_packed = nn.utils.rnn.pack_padded_sequence(
            embeded, lengths, batch_first=False)  # 因为传入的sentences不是batch_first
        # lstm_out最后一层的每一时刻 h_t, (seq_len, batch, num_directions * hidden_size)
        # h_n每一层的最后一个时刻 (num_layers * num_directions, batch, hidden_size)
        # h_n, c_n 不用手动做pad_packed_sequence处理，直接就是处理好的了，已经是(除去padding后的)有效值
        # h_n[-2, :, :] 是最后一层正向最后时刻，h_n[-1, :, :] 是最后一层反向最后时刻
        # h_n[0, :, :] 是第一层（最底层）的正向，h_n[1, :, :]是第一层（最底层）的反向
        # lstm_out[0].shape (9 * 10),这里的10是5*2的隐层向量; lstm_out[1] 是[3, 3, 2, 1]
        # h_n, c_n shape都是(6, 3, 5)，且h_n已经是除去pad后的有效的隐层
        lstm_out, (h_n, c_n) = self.lstm(embeded_packed)  # lstm_out仍然是扁平化的，所以想要用的话需要先pad_packed_sequence
        ###########
        # pad_packed_sequence是解压缩处理
        # lstm_out是tensor shape(seq_len, batch, num_directions * hidden_size); _lengths是len_list, 长度倒序
        # lstm_out中的后面padding位置是全零
        # 对于双向LSTM，lstm_out的长度是hidden_size*2, 即是某一个词的hidden由两部分组成，以sentence第一个词为例，是正向的第一个，反向的最后一个
        lstm_out, _lengths = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=False)  # 新的lstm_out shape(4, 3, 10); _lengths是[4, 3, 2]
        ############
        # 对于双向，多层的LSTM，想要得到句子表示，可以有两个方法
        # 1. 使用lstm_out
        # 2. 使用h_n
        ############ 以下为无attention版本
        if self.bidirectional:
            # h_n是每一层的最后一个时刻输出，shape(num_layers * num_directions, batch, hidden_size)
            # 具体排列顺序为第1层的正向隐层输出，第1层的反向隐层输出，第2层的正向，第2层的反向...
            # 所以h_n[-2]是最后一层的正向，h_n[-1]是最后一层的反向
            h_n = torch.cat((h_n[-2], h_n[-1]), 1)
        else:
            h_n = h_n[-1]
        y = self.output(h_n)
        probs = F.softmax(y, dim=1)
        return probs
        ############
        ############ 以上为无attention版本

        # ## 以下为attention版
        # if self.bidirectional:
        #     #attention representation, weights
        #     attn_r, attn_w = self.attn_bi(lstm_out)
        # else:
        #     attn_r, attn_w = self.attn(lstm_out)
        # y = self.output(attn_r)
        # probs = F.softmax(y, dim=1)
        # return probs, attn_w