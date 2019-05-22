#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-13 00:41:51
# @Author  : skyhiter

import argparse
import pickle

import nltk
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine, Events, State
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, Precision, Recall, RunningAverage
from ignite.utils import convert_tensor
from torchtext import data, vocab
from torchtext.data import BucketIterator, Iterator


class BatchGenerator:
    """BucketIterator/Iterator对象的进一步封装, 
    这里的目的是为了封装每条原始数据的index
    """

    def __init__(self, data_iter: Iterator, index: str, x_var: str, y_var: str):
        """param data_iter: data iterator, BucketIterator/Iterator
        param index: 数据原始index, int
        param x_var: 自变量, str
        param y_var: 因变量（标签）, str
        """
        self.data_iter, self.index, self.x_var, self.y_var = data_iter, index, x_var, y_var

    def __len__(self):
        return len(self.data_iter)

    def __iter__(self):
        for batch in self.data_iter:
            index = getattr(batch, self.index)  # 获取原始的index
            # 因为TEXT的include_lengths=True, 所以X是(indexed_sents, lengths)
            X = getattr(batch, self.x_var)  # 获取自变量x_var的数据
            Y = getattr(batch, self.y_var)  # 获取标签y_var的数据
            yield (index, X, Y)  # index保留的是原始的顺序，X是元组，Y是标签列表


def load_dataset(args):
    tokenizer = lambda sent: nltk.tokenize.word_tokenize(sent)  # NLTK提供的默认英语的tokenizer(就是分词的意思)，返回的是word的list
    # use_vocab默认是true, 这说明后面需要加入到vocab索引中；如果数据已经是数字形式标签(0和1)，则不需要再vocab化，所以LABEL需要use_vocab=False
    TEXT = data.Field(
        sequential=True, tokenize=tokenizer, lower=True, batch_first=False,
        include_lengths=True)  # include_lengths很有用，后面获取batch时会顺便返回实际长度
    LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)  # 或者直接使用data.LabelField(), is_target其实没啥大用
    fields = [('label', LABEL), ('review', TEXT)]  # label和review都是自己起的名字，对应数据文件的第一列和第二列
    test = data.TabularDataset(path='./data/_valid_saved.txt', format='TSV', fields=fields, skip_header=False)

    def add_index_attr(dataset):
        """给dataset强制添加index Field; 遍历每个Example对象，为index Field分配数值
        目的是后续shuffle时方便恢复到原顺序。函数顺便返回[[1, ['i', 'love', 'you']], ...]
        """
        _dataset_list = []
        dataset.fields['index'] = data.Field(sequential=False, use_vocab=False)  # 先给dataset的fields属性强加'index'字段
        for index, example in enumerate(dataset):  # 再给dataset的每个Example添加index值
            setattr(example, 'index', index)
            _dataset_list.append([example.label, example.review])
        return _dataset_list

    test_list = add_index_attr(test)  # 给test的每个example添加index，并返回list形式的dataset

    # 加载预存的stoi映射
    TEXT.build_vocab()
    with open('./data/pkl/stoi.pkl', 'rb') as f:
        stoi = pickle.load(f)
    TEXT.vocab.stoi = stoi

    _test_iter = Iterator(
        test,
        batch_size=args.test_batch,
        train=False,
        shuffle=False,
        sort=False,
        device=args.device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.review))

    test_iter = BatchGenerator(_test_iter, 'index', 'review', 'label')

    return test_iter, test_list


def load_model(args):
    with open('./data/saved_models/best_models_info.json', 'r') as f:
        best_models_info = eval(f.read())
    print("best model path: {}".format(best_models_info['best_model_path']))
    model = torch.load(best_models_info['best_model_path'], map_location=args.device)

    return model


def prepare_test_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    index, x, y = batch

    # index 不参与模型计算，所以不用convert
    return (index, convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def predict(args):
    test_iter, test_list = load_dataset(args)
    model = load_model(args)
    model.eval()

    pred_results = []
    with torch.no_grad():
        for batch in test_iter:
            index, x, y = prepare_test_batch(batch, device=device, non_blocking=False)
            y_probs = model(x)
            max_probs, preds = torch.max(y_probs, dim=1)
            index, max_probs, preds, y = map(lambda tensors: tensors.tolist(), \
                                            [index, max_probs, preds, y])
            pred_results.extend(list(zip(index, zip(max_probs, preds, y))))
    pred_results.sort(key=lambda item: item[0])
    assert len(pred_results) == len(test_list)
    print(pred_results[:10])

    with open('./results.txt', 'w') as f:
        f.write("prob\tpred\ttrue\ttext\n")
        for index, (prob, pred, true) in pred_results:
            # print(index, prob, pred)
            f.write("{:.3f}\t{}\t{}\t{}\n".format(prob, pred, true, ' '.join(test_list[index][1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction by pretraind model (Python 3.6 & Pytorch 1.1)")
    parser.add_argument("--test_batch", type=int, default=4, help="test batch size")
    parser.add_argument("--gpu", type=int, default=0, help="use GPU number, begin index 0")
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    setattr(args, "device", device)  # args 对象本没有device属性，这里强加进去，这样args就包含了更多的配置信息

    predict(args)
