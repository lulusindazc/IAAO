import argparse
from dataset.demo import DemoDataset
from dataset.object import ObjectDataset
from dataset.object_gs import ObjectgsDataset
import json

def update_args(args):
    config_path = f'configs/{args.config}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        setattr(args, key, config[key])
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str) 
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config', type=str, default='scannet')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    args = update_args(args)
    return args

def get_dataset(args):
    if args.dataset == 'demo':
        dataset = DemoDataset(args.seq_name)
    elif args.dataset == 'object':
        dataset = ObjectDataset(args.seq_name,args.obj_name)
    elif args.dataset == 'object_gs':
        dataset = ObjectgsDataset(args.seq_name,args.obj_name)
    else:
        print(args.dataset)
        raise NotImplementedError
    return dataset

