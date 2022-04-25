import os
import time

import numpy as np
import torch
import torch.optim

from data.datamgr import SimpleFewShotDataManager, CrossFewShotDataManager
from methods.backbone import model_dict
from methods.gnnnet import GnnNet
from options import parse_args, get_resume_file, load_warmup_state


# --- main function ---
if __name__=='__main__':

    # set numpy random seed
    np.random.seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parser argument
    params = parse_args('fs_test')


    for model_name in ["radar", "simulation"]:
        print("----------------------Model from %s----------------------" % model_name)
        for single in [True, False]:
            if single:
                few_shot_params = dict(n_way=params.n_way, n_support=params.n_shot, n_query=1)
                tep = params.tep * 3
            else:
                few_shot_params = dict(n_way=params.n_way, n_support=params.n_shot, n_query=12)
                tep = params.tep

            model = GnnNet(model_dict[params.model], **few_shot_params, leakyrelu=params.leakyrelu)
            model = model.cuda()
            state = torch.load('%s/%s/checkpoints/%s' % (params.save_dir, model_name, "best_model.tar"))['state']
            model.load_state_dict(state)
            model.eval()

            testset = "radar"
            for same_people in [True, False]:
                for test_name in ["seen_test", "unseen_test", "test"]:
                    test_datamgr = SimpleFewShotDataManager("%s/%s/%s.txt" % (params.data_dir, testset, test_name))
                    print("--------%s %s %s--------" % (test_name, 'same' if same_people else 'different',
                                                        'single' if single else 'multiple'))
                    test_loader = test_datamgr.get_data_loader(few_shot_params, tep=tep, group_by_people=True,
                                                               single=single, same_people= same_people)
                    model.test_loop(test_loader)
