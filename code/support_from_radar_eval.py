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

    for model_name in ["combination_train_transfer", "fine_tune_new", "combination_train_with_mmd_loss"]:
        if model_name == "checkpoints":
            continue
        print("----------------------Model from %s----------------------" % model_name)
        model = GnnNet(model_dict[params.model], **few_shot_params, leakyrelu=params.leakyrelu)
        model = model.cuda()
        state = torch.load('%s/%s/checkpoints/%s' % (params.save_dir, model_name, "best_model.tar"))['state']
        model.load_state_dict(state)
        model.eval()
        if params.strategy == 'combination':
            eval_func = model.combination_system_evaluation
        elif params.strategy == 'sliding window':
            eval_func = model.sliding_window_system_evaluation
        elif params.strategy == 'contest':
            eval_func = model.contest_system_evaluation
        else:
            raise NotImplementedError

        # for same_people in [True, False]:
        #     for env in ["env%d_eval" % i for i in range(1, 6)]:
        #         datamgr= SystemDataManager("%s/eval/%s.txt" % (params.data_dir, env),
        #                                    "%s/eval/%s.txt" % (params.data_dir, env))
        #         print("-------Eval in %s with %s people-------" % (env, "same" if same_people else "different"))
        #         dataloader = datamgr.get_data_loader(few_shot_params, tep=params.tep, same_people=same_people)
        #
        #         avg_acc = eval_func(dataloader)
        #
        #         print('--- Eval in %s with %s people | Acc = %4.2f%% ---' %
        #               (env, "same" if same_people else "different", avg_acc * 100))

        for same_people in [False]:
            for i in range(2, 6):
                accs = []
                for j in range(2, 6):
                    if i == j:
                        continue
                    datamgr= SystemDataManager("%s/eval/env%d_eval.txt" % (params.data_dir, j),
                                               "%s/eval/env%d_eval.txt" % (params.data_dir, i))
                    print("-------Eval in env%d using env%d with %s people-------" %
                          (i, j, "same" if same_people else "different"))
                    dataloader = datamgr.get_data_loader(few_shot_params, tep=params.tep, same_people=same_people)

                    avg_acc = eval_func(dataloader)
                    accs.append(avg_acc)
                    print("----- Eval in env%d using env%d with %s people | Acc = %4.2f%% -----" %
                          (i, j, "same" if same_people else "different", avg_acc * 100))
                print("--- Eval in env%d using different environments with %s people | Acc = %4.2f%% -----"%
                          (i, "same" if same_people else "different", np.mean(accs) * 100))
