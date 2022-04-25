import numpy as np
import os
import glob
import torch
import argparse

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' % script)
    parser.add_argument('--name'        , default='combination_train_transfer_with_mmd_loss_before_gnn', type=str, help='')

    parser.add_argument('--model', default='ResNet10', help='model: Conv{4|6} / ResNet{10|18|34}') # we use ResNet10 in the paper
    parser.add_argument('--method', default='gnnnet',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/gnnnet')
    parser.add_argument("--leakyrelu", default=True)

    parser.add_argument('--n_way' , default=3, type=int,  help='class num to classify')
    parser.add_argument('--n_shot'      , default=3, type=int,  help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query'      , default=15, type=int,  help='number of sample to be query in each class')

    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
    parser.add_argument('--save_dir'    , default='./output', type=str, help='')
    parser.add_argument('--data_dir'    , default='./filelists', type=str, help='')
    parser.add_argument('--lambda_0', default=10, type=float,
                        help='The hyper-parameter \lambda_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
                             'We adopt 1 for DenseNets and 7.5 for ResNets and ResNeXts, except for using 5 for ResNet-101.')
    if script == 'train':
        parser.add_argument('--trainset', default='radar_fine_tune')
        parser.add_argument('--epochs'   , default=200, type=int, help='number of train epochs')
        parser.add_argument('--tep'   , default=50, type=int, help='number of few shot tasks in each epoch')
        parser.add_argument('--save_freq'   , default=25, type=int, help='Save frequency')
        parser.add_argument('--resume'      , default='', type=str, help='continue from previous trained model with largest epoch')
        parser.add_argument('--resume_epoch', default=-1, type=int, help='')
        parser.add_argument('--warmup'      , default='baseline', type=str, help='continue from baseline, neglected if resume is true')
    elif script == 'fs_test':
        parser.add_argument('--tep'   , default=500, type=int, help='number of few shot tasks in each epoch')
        parser.add_argument('--save_epoch', default = -1, type=int,help ='load the model trained in x epoch, use the best model if x is -1')
    elif script == 'system_eval':
        parser.add_argument('--tep', default=2000, type=int, help='number of few shot tasks in each epoch')
        parser.add_argument('--save_epoch', default=-1, type=int,
                            help='load the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--strategy', default='contest', type=str)
    elif script == 'fine_tune':
        parser.add_argument('--tep'   , default=50, type=int, help='number of few shot tasks in each epoch')
        parser.add_argument('--resume'      , default='simulation', type=str, help='continue from previous trained model with largest epoch')
        parser.add_argument('--resume_epoch', default=-1, type=int, help='')
        parser.add_argument('--fine_tune_dataset', default='radar_fine_tune')
        parser.add_argument('--epochs'   , default=220, type=int, help='number of train epochs')
        parser.add_argument('--save_freq'   , default=5, type=int, help='Save frequency')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir, resume_epoch=-1):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    epoch = max_epoch if resume_epoch == -1 else resume_epoch
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def load_warmup_state(filename, method):
    print('  load pre-trained model file: {}'.format(filename))
    warmup_resume_file = get_resume_file(filename)
    tmp = torch.load(warmup_resume_file)
    if tmp is not None:
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if 'relationnet' in method and "feature." in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif method == 'gnnnet' and 'feature.' in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
                newkey = key.replace("feature.","")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
    else:
        raise ValueError(' No pre-trained encoder file found!')
    return state

