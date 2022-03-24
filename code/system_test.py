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
    random.seed(99)
    np.random.seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parser argument
    params = parse_args('test')
    print(params)

    few_shot_params = dict(n_way = params.n_way, n_support = params.n_shot, n_query = 1)

    for model_name in ["radar", "simulation"]:
    # for model_name in ["simulation_train_fine_tune"]:
        print("----------------------Model from %s----------------------" % model_name)
        model = GnnNet(model_dict[params.model], **few_shot_params, leakyrelu=params.leakyrelu)
        model = model.cuda()
        state = torch.load('%s/%s/checkpoints/%s' % (params.save_dir, model_name, "best_model.tar"))['state']
        model.load_state_dict(state)
        model.eval()

        for support_files in ["radar", "simulation"]:
            datamgr= SystemDataManager("%s/%s/%s.txt" % (params.data_dir, support_files, "test"),
                                 "%s/%s/%s.txt" % (params.data_dir, "radar", "test"))
            for same_people in [True, False]:
                print("-------Dataset from %s (%s)-------" % (support_files, "same" if same_people else "different"))
                dataloader = datamgr.get_data_loader(few_shot_params, tep=params.tep, same_people=same_people)

                avg_acc = model.system_evaluation(dataloader)
                print('--- Dataset: %s (%s people) | Acc = %4.2f%% ---' %
                      (support_files, "same" if same_people else "different", avg_acc * 100))
