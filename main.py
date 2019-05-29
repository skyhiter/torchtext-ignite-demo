#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-09 20:29:55
# @Author  : skyhiter

import argparse
import inspect
import json
import pickle
import platform
import sys
from pathlib import Path
from pprint import pprint

import ignite
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Events, create_supervised_evaluator, create_supervised_trainer)
from ignite.handlers import (EarlyStopping, ModelCheckpoint, TerminateOnNan, Timer)
from ignite.metrics import Accuracy, Loss, Precision, Recall, RunningAverage

from data_loader import load_dataset
from log import logger
from nnet.lstm import LSTMClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch classification Project (Python 3.6 & Pytorch 1.0)")
    parser.add_argument("--bi", action="store_true", default=True, help="bidirectory LSTM")
    parser.add_argument("--nclass", type=int, default=2, help="number of classes to predict")
    parser.add_argument("--nhid", type=int, default=50, help="size of RNN hidden layer")
    parser.add_argument("--nlayers", type=int, default=1, help="number of layers of LSTM")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epoch")
    parser.add_argument("--log_interval", type=int, default=100, help="log result per interval interval")
    parser.add_argument("--batch_size", type=int, default=32, help="train batch size")
    parser.add_argument("--valid_batch", type=int, default=64, help="valid batch size")
    parser.add_argument("--test_batch", type=int, default=64, help="test batch size")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--early_stop", type=int, default=20, help="max time step of answer sequence")
    parser.add_argument("--seed", type=int, default=123456, help="random seed for reproduction")
    parser.add_argument("--gpu", type=int, default=0, help="use GPU number, begin index 0")
    parser.add_argument("--embd", type=str, default="glove.6B.50d", help="use which pretrained embedding")
    args = parser.parse_args()

    return args


def get_resource(args):
    platform_info = platform.platform()
    logger.debug("current platform: {}".format(platform_info))
    home_root = Path.home()  # new in python 3.5
    # local machine(macOS)
    # macOS may print 'Darwin-18.3.0-x86_64-i386-64bit'
    if "darwin" in platform_info.lower():
        embd_path_refix = Path(home_root / "doc/dataset/embedding")
    # remote server(Centos)
    # Centos may print 'Linux-3.12.0-292.01.5.el2.x86_64-x86_64-with-centos-7.3.1223-Core'
    elif "centos" in platform_info.lower():
        embd_path_refix = Path(home_root / "dataset/embedding")
    else:
        logger.error("Error! platform info is {}".format(platform_info))
    embd = {
        "weibo": "word2vec/weibo875_cnt5_dim100.word2vec",
        "glove.6B.50d": "glove/glove.6B.50d.txt",
        "glove.6B.100d": "glove/glove.6B.100d.txt",
        "glove.6B.200d": "glove/glove.6B.200d.txt",
        "glove.6B.300d": "glove/glove.6B.300d.txt"
    }

    return embd_path_refix / embd[args.embd]


def run(args):
    train_iter, valid_iter, test_iter, indexed_vector = load_dataset(args)
    # iters_per_epoch = len(train_iter) // 100 * 100  # 取整百，比如train dataset是7463,batch16，则每epoch有466.4 iteration
    iters_per_epoch = len(train_iter)

    model = LSTMClassifier(
        indexed_vector,
        hidden_dim=args.nhid,
        output_dim=args.nclass,
        num_layers=args.nlayers,
        dropout=args.dropout,
        bidirectional=args.bi)

    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()))
    criterion = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=criterion, device=args.device)
    train_evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'accuracy': Accuracy(),
            'precision': Precision(),
            'recall': Recall(),
            'loss': Loss(criterion)
        },
        device=args.device)
    valid_evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'accuracy': Accuracy(),
            'precision': Precision(),
            'recall': Recall(),
            'loss': Loss(criterion)
        },
        device=args.device)

    def loss_score(engine):
        loss = engine.state.output
        return -loss  # 分数越高越好，所以loss取负

    def acc_score(engine):
        accuracy = engine.state.metrics['accuracy']
        return accuracy

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        train_iter_num = engine.state.iteration
        logger.info("Epoch {} Iteration {}: Loss {:.4f}"
                    "".format(engine.state.epoch, train_iter_num, engine.state.output))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_validation_results(engine):
        train_iter_num = engine.state.iteration
        if train_iter_num > iters_per_epoch and train_iter_num % args.log_interval == 0:
            valid_evaluator.run(valid_iter)
            metrics = valid_evaluator.state.metrics
            logger.info(
                "Validation Results - Epoch {}, Iter {}: Avg accuracy {}, Precision {}, Recall {}, valid loss {:.4f}"
                "".format(engine.state.epoch, train_iter_num, metrics['accuracy'], metrics['precision'].tolist(),
                          metrics['recall'].tolist(), metrics['loss']))

    # train的每ITERATION检查loss是否是 "Nan"
    # 是的话终止训练
    terminateonnan = TerminateOnNan(output_transform=lambda output: output)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, terminateonnan)

    checkpoint_handler = ModelCheckpoint(
        dirname='./data/saved_models/',
        filename_prefix='checkpoint',
        score_function=acc_score,
        score_name="acc",
        save_interval=None,  # 按次数周期保存
        n_saved=3,
        require_empty=False,  # 强制覆盖
        create_dir=True,
        save_as_state_dict=False)
    # 因为valid的epoch往往是1，所以Events.EPOCH_COMPLETED和Events.COMPLETED是一样的
    valid_evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'model': model})
    patience = int(args.early_stop * (iters_per_epoch / args.log_interval))
    earlystop_handler = EarlyStopping(patience=patience, score_function=acc_score, trainer=trainer)
    earlystop_handler._logger = logger
    valid_evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)

    trainer_bar = ProgressBar()
    trainer_bar.attach(trainer, output_transform=lambda x: {'loss': x})

    trainer.run(data=train_iter, max_epochs=args.epochs)
    logger.info("Best model: Epoch {}, Train iters {}, Valid iters {}, acc {}"
                "".format(earlystop_handler.best_state['epoch'], \
                          earlystop_handler.best_state['iters'], \
                          earlystop_handler.best_state['valid_iters'],
                          earlystop_handler.best_score))
    logger.info("Valid results in best model: {}".format(earlystop_handler.best_state['metrics']))
    logger.info("Best models info: {}".format(str(checkpoint_handler._saved)))  # [(0.65,['model_6_acc=0.65.pth']),...]
    best_models_info = {
        'model_args': str(args.__dict__),
        'checkpint_saved': checkpoint_handler._saved,
        'train_epoch': earlystop_handler.best_state['epoch'],
        'train_iters': earlystop_handler.best_state['iters'],
        'valid_iters': earlystop_handler.best_state['valid_iters'],
        'best_model_path': checkpoint_handler._saved[-1][1][0],  #checkpoint_handler._saved按sore升序排列的
        'best_score': checkpoint_handler._saved[-1][0],
        'score_function': checkpoint_handler._score_function.__name__,
        'valid_results': {
            'accuracy': earlystop_handler.best_state['metrics']['accuracy'],
            'precision': earlystop_handler.best_state['metrics']['precision'].tolist(),
            'recall': earlystop_handler.best_state['metrics']['recall'].tolist(),
            'loss': earlystop_handler.best_state['metrics']['loss']
        }
    }
    print(checkpoint_handler._saved)
    pprint(str(best_models_info))
    # exit()
    # with open('./data/pkl/best_models_path.pkl', 'wb') as f:
    #     pickle.dump(list(map(lambda model_info: model_info[1][0], checkpoint_handler._saved)), f)
    with open('./data/saved_models/best_models_info.json', 'w') as f:
        f.write(repr(best_models_info))

    def test(test_iter, args):
        # with open('./data/pkl/best_models_path.pkl', 'rb') as f:
        #     best_models = pickle.load(f)
        with open('./data/saved_models/best_models_info.json', 'r') as f:
            best_models_info = eval(f.read())
        print("best models info: {}".format(best_models_info))
        logger.info("best model path: {}".format(best_models_info['best_model_path']))
        model = torch.load(best_models_info['best_model_path'], map_location=args.device)
        test_evaluator = create_supervised_evaluator(
            model=model,
            metrics={
                'accuracy': Accuracy(),
                'precision': Precision(),
                'recall': Recall(),
                'loss': Loss(criterion)
            },
            device=args.device)

        @test_evaluator.on(Events.COMPLETED)
        def log_test_results(engine):
            metrics = engine.state.metrics
            logger.info("Test Results: Avg accuracy: {}, Precision: {}, Recall: {}, Loss: {}"
                        "".format(
                                  metrics['accuracy'], \
                                  metrics['precision'].tolist(),
                                  metrics['recall'].tolist(),
                                  metrics['loss']
                            )
                        )

        test_evaluator.run(test_iter)

    test(valid_iter, args)


def init():
    def mkdir():
        Path('./data/pkl').mkdir(exist_ok=True, parents=True)  # if not exists, mkdir
        Path('./data/saved_models').mkdir(exist_ok=True, parents=True)  # if not exists, mkdir
        Path('./data/logs').mkdir(exist_ok=True, parents=True)  # if not exists, mkdir
        Path('./data/vector_cache').mkdir(exist_ok=True, parents=True)  # if not exists, mkdir

    mkdir()
    logger.info("Command to run: " + " ".join(sys.argv))
    args = parse_args()
    logger.debug("Parsed args: " + str(args))
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    setattr(args, "device", device)  # args 对象本没有device属性，这里强加进去，方便后面只传args
    embedding_path = get_resource(args)  # embedding_path是Path对象
    setattr(args, "embd", embedding_path)  # args中的embd更新为实际具体的路径，此时args.embd是Path对象
    logger.info("Parsed args updated: " + str(args))

    torch.manual_seed(args.seed)  # 手动设置非cuda的随机数种子
    torch.cuda.manual_seed_all(args.seed)  # 如果没有cuda设备,会自动忽略该函数，不会发生错误

    return args


if __name__ == '__main__':
    args = init()
    run(args)
