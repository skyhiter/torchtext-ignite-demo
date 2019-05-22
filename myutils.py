#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-05 10:47:46
# @Author  : skyhiter

import pickle


def pkl_dump(your_object, des_file):
    with open(des_file, 'wb') as f:
        pickle.dump(your_object, f)


def pkl_load(pkl_file):
    with open(pkl_file, 'rb') as f:
        your_object = pickle.load(f)
    return your_object