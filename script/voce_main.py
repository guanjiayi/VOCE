import os.path as osp
import wandb
import yaml
from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

from safe_rl.runner import Runner

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")
EXP_NAME_KEYS = {"epochs": "epoch"}
DATA_DIR_KEYS = {"cost_limit": "cost"}

class Offlinebuffer(Dataset):
    """Offline dataset load class."""
    def __init__(self,based_dir:str):
        """
            Load the offline dataset from the local dir.
            Args:
                based_dir: the local dir of the offline dataset.
        """
        offlinedata = h5py.File(based_dir,"r")
        self.state = offlinedata['observation'][:]
        self.next_state = offlinedata['next_observation'][:]
        self.action = offlinedata['action'][:]
        self.reward = offlinedata['reward'][:]
        self.not_done = offlinedata['done'][:]
        self.cost = offlinedata['cost'][:]
        offlinedata.close()

    def get_buffer(self,batch_size):
        """
            Get the batch of buffer in tensor type.
            Args:
                batch_size: the size of the get the batch.
            Return:
                The tensor contains the sate, action, next_state, reward, mask.
        """
        ind = np.random.randint(0,len(self.state),size=batch_size)
        return(
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.cost[ind]),
            torch.FloatTensor(self.not_done[ind]),
            # torch.FloatTensor(self.steps[ind])
        )
    
    def get_sample(self,batch_size):
        """
            Get the batch of buffer in tensor type.
            Args:
                batch_size: the size of the get the batch.
            Return:
                The tensor contains the sate, action, next_state, reward, mask.
        """
        ind = np.random.randint(0,len(self.state),size=batch_size)
        data = {}
        data['obs']=torch.FloatTensor(self.state[ind])
        data['act']=torch.FloatTensor(self.action[ind])
        data['rew']=torch.FloatTensor(self.reward[ind])
        data['obs2']=torch.FloatTensor(self.next_state[ind])
        data['done']=torch.FloatTensor(self.not_done[ind])
        data["cost"]=torch.FloatTensor(self.cost[ind])
        return data


def load_config(config_path="default_config.yaml") -> dict:
    '''
    Load the config parameters from the yaml file.
    '''
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def gen_exp_name(config: dict, suffix=None):
    suffix = "" if suffix is None else "_" + suffix
    name = config["policy"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + str(config[k])
    return name + suffix

def gen_data_dir_name(config: dict):
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-PointButton1-v0')
    # parser.add_argument('--policy', '-p', type=str, default='cvpo')
    parser.add_argument('--pretrain_dir', '-pre', type=str, default=None)
    parser.add_argument('--load_dir', '-d', type=str, default=None)
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--seed', '-s', type=int, default=8)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--suffix', '--id', type=str, default=None)
    parser.add_argument('--no_render', action="store_true")
    parser.add_argument('--sleep', type=float, default=0.003)
    parser.add_argument('--wandb','-wan',action='store_true',help='Flag for logging data via wandb')
    parser.add_argument('--datadir',type=str,default='../buffer/Safexp-PointButton1-v0-08.hdf5')
    parser.add_argument('--offline','-offline',action='store_true',default='offline train flag')
    args = parser.parse_args()

    # Transform to the dict type
    args_dict = vars(args)

    config_path = osp.join(CONFIG_DIR, "config_voce.yaml")

    config = load_config(config_path)
    config.update(args_dict)

    config["exp_name"] = gen_exp_name(config, args.suffix)
    config["data_dir"] = gen_data_dir_name(config)

    # Init the wandb
    if config['wandb']:
        project_name = 'voce'+str('_')+config['env']
        run = wandb.init(project=project_name)

    datasets = Offlinebuffer(based_dir=config['datadir'])

    trainer = Runner(**config)

    if args.mode == 'train':
        trainer.train(datasets)
    else:
        trainer.eval(render=not args.no_render, sleep=args.sleep)

    if config['wandb']:
        wandb.finish()
    print('success!')
