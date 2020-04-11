from transformers import BertConfig, BertTokenizer
from model import SQuAD_Bert
import random
import numpy as np
import torch
import logging

def set_seed(args):         # 对所有可能出现随机数的部分设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def compute_f1(actual_match_preds, actual_real_match_span, all_pos_num):
    FN = 0
    TN = 0
    TP = 0
    FP = 0
    for match, real_match, pos_num in zip(actual_match_preds, actual_real_match_span, all_pos_num):
        match_start = match[0]
        match_end = match[1]
        real_match_start = real_match[0]
        real_match_end = real_match[1]
        if match_end < real_match_start or match_start > real_match_end:
            tp = 0
            fp = match_end - match_start + 1
            tn = real_match_end - real_match_start + 1
            fn = pos_num - tp - fp - tn
            FN += fn
            TN += tn
            TP += tp
            FP += fp
        elif match_end <= real_match_end <= real_match_start <= match_start:
            tp = real_match_start - real_match_end + 1
            fp = 0
            tn = real_match_end - match_end + match_start -real_match_start
            fn = pos_num - tp - fp - tn
            FN += fn
            TN += tn
            TP += tp
            FP += fp
        elif real_match_end <= match_end <= match_start <= real_match_start:
            tp = real_match_start - real_match_end + 1
            fp = match_end - real_match_end + real_match_start -match_start
            tn = 0
            fn = pos_num - tp - fp - tn
            FN += fn
            TN += tn
            TP += tp
            FP += fp
        elif match_end <= real_match_end <= match_start <= real_match_start:
            tp = match_start - real_match_end + 1
            fp = real_match_end - match_end
            tn = real_match_start - match_start
            fn = pos_num - tp - fp - tn
            FN += fn
            TN += tn
            TP += tp
            FP += fp
        elif real_match_end <= match_end <= real_match_start <= match_start:
            tp = real_match_start - match_end + 1
            fp = match_start - real_match_start
            tn = match_end - real_match_end
            fn = pos_num - tp - fp - tn
            FN += fn
            TN += tn
            TP += tp
            FP += fp
        else:
            pass
    return 2*TP/(2*TP + FP + FN)