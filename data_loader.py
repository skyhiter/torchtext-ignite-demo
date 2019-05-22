#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-15 22:04:10
# @Author  : skyhiter

import logging
import pickle
import random
from itertools import islice  # 仅用于读文件时跳过文件第一行
from pathlib import Path

import nltk
import torch
from torchtext import data, vocab
from torchtext.data import BucketIterator, Iterator

from log import logger
from myutils import pkl_dump, pkl_load


class BatchGenerator:
    """BucketIterator/Iterator对象的进一步封装, 这样可以独立于各个TEXT字段名;
    其实直接使用BucketIterator/Iterator对象自带的batch也行
    """

    def __init__(self, data_iter: Iterator, x_var: str, y_var: str):
        """param data_iter: data iterator, BucketIterator/Iterator
        param x_var: 自变量, str
        param y_var: 因变量（标签）, str
        """
        self.data_iter, self.x_var, self.y_var = data_iter, x_var, y_var

    def __len__(self):
        return len(self.data_iter)

    def __iter__(self):
        for batch in self.data_iter:
            X = getattr(batch, self.x_var)  # 获取自变量x_var的数据, X是[sentences, lengths]
            Y = getattr(batch, self.y_var)  # 获取标签y_var的数据
            yield (X, Y)


def load_dataset(args):
    tokenizer = lambda sent: nltk.tokenize.word_tokenize(sent)  # NLTK提供的默认英语的tokenizer(就是分词的意思)，返回的是word的list
    # use_vocab默认是true, 这说明后面需要加入到vocab索引中；如果数据已经是数字形式标签(0和1)，则不需要再vocab化，所以LABEL需要use_vocab=False
    TEXT = data.Field(
        sequential=True, tokenize=tokenizer, lower=True, batch_first=False,
        include_lengths=True)  # include_lengths很有用，后面获取batch时会顺便返回实际长度
    LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)  # 或者直接使用data.LabelField(), is_target其实没啥大用
    fields = [('label', LABEL), ('review', TEXT)]  # label和review都是自己起的名字，对应数据文件的第一列和第二列
    # 若split_ratio是(train, test, valid)的顺序，返回值是元组(train, valid, test)
    # 若split_ratio是float则表示train:valid=ratio:(1-ratio)，返回值是元组(train, valid)
    train, valid, test = data.TabularDataset(
        path='./data/MR_10662.txt', \
        format='TSV', fields=fields, \
        skip_header=False).split(split_ratio=[0.7, 0.1, 0.2], random_state=random.seed(args.seed)) # split函数里用到了shuffle，所以会先打乱数据集再划分
    # 可以通过train[0].review访问第一条数据的review列

    dataset_lengths = [len(train), len(valid), len(test)]
    logger.info("total data size {}. train size {}, valid size {}, test size {}".format(
        sum(dataset_lengths), *dataset_lengths))

    # 为方便后续调试，现将切分的train和valid、test写入文件保存，方便后面肉眼debug
    # 在data.TabularDataset.split()时顺序已经打乱了
    dataset_name = {'train': train, 'valid': valid, 'test': test}
    for name, dataset in dataset_name.items():
        path = Path('./data/_{}_saved.txt'.format(name))
        if path.exists():
            continue
        with open(path, 'w') as f:
            for item in dataset:
                f.write("{}\t{}\n".format(item.label, ' '.join(item.review)))

    # build_vocab 比较费时间，可以把结果TEXT.vocab先存起来
    # 如果之前没有保存TEXT.vocab，则需要先TEXT.build_vocab
    if not Path('./data/pkl/TEXT.vocab').exists():
        pretrained_vector = vocab.Vectors(
            name=args.embd,  # 预训练词向量，第一行的(vocab, dim)带不带都行，会自动识别\
            cache='./data/vector_cache',  # 缓存路径，torchtext会将预训练词向量转成二进制形式暂存\
            max_vectors=None)
        # 想看'the'的词向量: TEXT.vocab.vectors[TEXT.vocab.stoi['the']
        TEXT.build_vocab(
            train, valid, \
            max_size=None, \
            vectors=pretrained_vector)
        # 之后可以TEXT.vocab.itos[0]或TEXT.vocab.stoi['<unk>']查看vocab和index的映射
        # 也可查看前10个词频最高的词 TEXT.vocab.freqs.most_common(10), 比如
        # 前10词频[('.', 12572), ('the', 9069), (',', 9023), ('a', 6569), ('and', 5594), ('of', 5461), ('to', 3819), ("'s", 3201), ('is', 3200), ('it', 3075)]

        # 把TEXT.vocab存起来，下一次就不用TEXT.build_vocab
        pkl_dump(TEXT.vocab, './data/pkl/TEXT.vocab')

        # 将两个映射保存，以便后面预测时使用
        pkl_dump(TEXT.vocab.stoi, './data/pkl/stoi.pkl')
        pkl_dump(TEXT.vocab.itos, './data/pkl/itos.pkl')
    else:
        TEXT.vocab = pkl_load(Path('./data/pkl/TEXT.vocab'))

    # indexed_vector size= (max+2, 50) (+2: <unk> and <pad>; 50: glove vector dim 50)
    indexed_vector = TEXT.vocab.vectors  # 已经按照vocab的索引顺序重新排列的vectors, 0是unk，1是padding，剩下的先按词频降序再按字典序排列
    #*是解包意思，即将list转为独立变量，.运算符优先级很高，所以先计算.shape再解包
    logger.info('vocab size {}, embedding dim {}'.format(*indexed_vector.shape))

    # _train_iter的train标记为True, 每个epoch后都会自动shuffle，即每个epoch都不一样
    # _valid_iter的train标记为False, 每个epoch都是一样的顺序（未必跟valid原顺序一致，因为为了最小padding原则进行了调整）
    _train_iter, _valid_iter = BucketIterator.splits(
        (train, valid),  # 元组的第一个必须是train，这是规定；元组第0个的train标记设置为True，其他都为False
        batch_sizes=(args.batch_size, args.valid_batch),
        device=args.device,
        sort_key=lambda x: len(x.review),
        shuffle=True,
        sort_within_batch=True)
    # test集合总是保持原顺序，甚至跟原数据(test.txt)顺序都一样（不shuffle，不sort），即使进行多个epoch也这样
    _test_iter = Iterator(
        test,
        batch_size=args.test_batch,
        train=False,
        shuffle=False,
        sort=False,
        device=args.device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.review))

    train_iter = BatchGenerator(_train_iter, "review", "label")
    valid_iter = BatchGenerator(_valid_iter, "review", "label")
    test_iter = BatchGenerator(_test_iter, "review", "label")

    return train_iter, valid_iter, test_iter, indexed_vector


if __name__ == '__main__':
    tokenizer = lambda sent: nltk.tokenize.word_tokenize(sent)
    print(tokenizer('simplistic , silly and tedious .'))
