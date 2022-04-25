import os
import random
import time

import numpy as np
import torch
import torch.optim

from data.datamgr import *
from methods.backbone import model_dict
from methods.gnnnet import GnnNet
from options import parse_args, get_resume_file, load_warmup_state


# --- main function ---
if __name__=='__main__':

    # set numpy random seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parse argument
    params = parse_args('system_eval')
    print("System Evaluation Strategy:", params.strategy)

    few_shot_params = dict(n_way = params.n_way, n_support = params.n_shot, n_query = 1)

    for model_name in ["radar"]:
        print("----------------------Model from %s----------------------" % model_name)
        model = GnnNet(model_dict[params.model], **few_shot_params, leakyrelu=params.leakyrelu)
        model = model.cuda()
        state = torch.load('%s/%s/checkpoints/%s' % (params.save_dir, model_name, "best_model.tar"))['state']
        model.load_state_dict(state)
        model.eval()

        for i in range(5, 12):
            datamgr= SystemDataManager("%s/eval/env5_activity_num/%d_class_eval.txt" % (params.data_dir, i),
                                       "%s/eval/env5_activity_num/%d_class_eval.txt" % (params.data_dir, i))
            for same_people in [True, False]:
                print("-------%d classes eval in env5 (%s people)-------" % (i, "same" if same_people else "different"))
                dataloader = datamgr.get_data_loader(few_shot_params, tep=params.tep, same_people=same_people)

                if params.strategy == 'combination':
                    avg_acc = model.combination_system_evaluation(dataloader)
                elif params.strategy == 'sliding window':
                    avg_acc = model.sliding_window_system_evaluation(dataloader)
                elif params.strategy == 'contest':
                    avg_acc = model.contest_system_evaluation(dataloader)
                else:
                    raise NotImplementedError

                print('--- %d classes eval in env5 (%s people) | Acc = %4.2f%% ---' %
                      (i, "same" if same_people else "different", avg_acc * 100))
