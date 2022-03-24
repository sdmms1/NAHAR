import os
import time

import numpy as np
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from data.datamgr import *
from methods.backbone import model_dict
from methods.gnnnet import GnnNet
from options import parse_args, get_resume_file, load_warmup_state


# --- main function ---
if __name__=='__main__':

    # set numpy random seed
    np.random.seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parser argument
    params = parse_args('fine_tune')
    print('--- baseline training: %s ---\n' % params.name)
    print("[%s] Fine Tune Dataset: %s" % (params.name, params.fine_tune_dataset))

    # output and tensorboard dir
    params.tf_dir = '%s/%s/tb/'% (params.save_dir, params.name)
    if not os.path.isdir(params.tf_dir):
        os.makedirs(params.tf_dir)
    params.checkpoint_dir = '%s/%s/checkpoints'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- prepare dataloader ---')

    few_shot_params    = dict(n_way = params.n_way, n_support = params.n_shot, n_query = params.n_query)

    base_datamgr = FineTuneDataManager(os.path.join(params.data_dir, params.fine_tune_dataset, "ft_support.txt"),
                                     os.path.join(params.data_dir, params.fine_tune_dataset, "ft_query.txt"))
    base_loader = base_datamgr.get_data_loader(few_shot_params, tep = params.tep)
    val_datamgr = FineTuneDataManager(os.path.join(params.data_dir, params.fine_tune_dataset, "val_support.txt"),
                                     os.path.join(params.data_dir, params.fine_tune_dataset, "val_query.txt"))
    val_loader = base_datamgr.get_data_loader(few_shot_params, tep = params.tep)

    # model
    print('\n--- build model ---')
    model = GnnNet(model_dict[params.model], **few_shot_params, leakyrelu=params.leakyrelu)
    model = model.cuda()

    # load model
    start_epoch = 0
    if params.resume != '':
        resume_file = get_resume_file('%s/%s/checkpoints/'%(params.save_dir, params.resume), params.resume_epoch)
        assert resume_file is not None
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch']+1
        model.load_state_dict(tmp['state'])
        print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
    else:
        print('  warm up with pretrain baseline network')
        state = load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup), params.method)
        model.feature.load_state_dict(state, strict=False)

    # training
    print('\n--- start the training ---')

    tb = SummaryWriter(log_dir=params.tf_dir) if params.tf_dir is not None else None

    # get optimizer and checkpoint path
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)

    # for validation
    max_acc = 0
    total_it = 1

    # start
    for epoch in range(start_epoch, params.epochs):
        model.train()
        train_loss = model.train_loop(epoch, base_loader,  optimizer) # model are called by reference, no need to return

        model.eval()
        test_loss, test_acc = model.test_loop(val_loader)

        if tb:
            tb.add_scalar("Test Acc", test_acc, epoch + 1)
            tb.add_scalars("Loss", {"Train": train_loss, "Test": test_loss}, epoch + 1)

        if test_acc > max_acc :
            print("best model! save...")
            max_acc = test_acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        else:
            print("GG! best accuracy {:f}".format(max_acc))

        if ((epoch + 1) % params.save_freq == 0) or (epoch == params.epochs-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
