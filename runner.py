import argparse
import shutil
import ray
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import tensorflow as tf
from config import config as envconf
from rldock.environments.lactamase import LactamaseDocking
from rnntest import MyKerasRNN
import copy
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-c', type=str, required=True)
    return parser.parse_args()

checkpoint = "/Users/austin/checkpoint_726/checkpoint-726tmu"

def env_creator(env_config):
    return LactamaseDocking(env_config)
    # return an env instance


if __name__ == '__main__':
    register_env("lactamase_docking", env_creator)

    ray.init()
    # memory_story = 256.00  * 1e+9
    # obj_store = 128.00 * 1e+9
    # ray.init(memory=memory_story, object_store_memory=obj_store)
    ModelCatalog.register_custom_model("rnn", MyKerasRNN)


    envconf['normalize'] = False
    # envconf['protein_wo_ligand'] = args.i
    # envconf['oe_box'] = None
    envconf['random'] = False
    envconf['random_dcd'] = False
    envconf['debug'] = False


    d = {
        "env": 'lactamase_docking',
        'log_level': "INFO",
        "env_config": envconf,
        "gamma": 0.95,
        'shuffle_sequence' : False,
        'eager': False,
        'reuse_actors' : False,
        "num_gpus": 0,
        "train_batch_size": 64,
        "sample_batch_size": 64,
        'sgd_minibatch_size': 32,
        "num_workers":  1,
        "num_envs_per_worker": 1,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 16,
        "vf_loss_coeff": 5e-2,
       'vf_share_layers' : True,
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 8,
        }}
    ppo_config = ppo.DEFAULT_CONFIG
    ppo_config.update(d)

    get_dock_marks = []

    workers = RolloutWorker(env_creator, ppo.PPOTFPolicy, env_config=envconf, policy_config=d)
    with open(checkpoint, 'rb') as c:
        c = c.read()
        c = pickle.loads(c)
        print(list(c.keys()))
        workers.restore(c['worker'])
    fp_path = "/Users/austin/PycharmProjects/RLDock/"
    with open("log.pml", 'w') as fp:
        with open("test.pml", 'w') as f:
            for j in range(2):
                rs = workers.sample()
                print(rs)
                print(list(rs.keys()))
                ls = rs['actions'].shape[0]
                for i in range(ls):
                    i += j * ls
                    ligand_pdb = rs['infos'][i - j *ls]['atom']
                    protein_pdb_link = rs['infos'][i - j * ls]['protein']

                    with open(fp_path + 'pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
                        f.write(ligand_pdb)
                    shutil.copyfile(protein_pdb_link, fp_path + 'pdbs_traj/test_p' + str(i) + '.pdb')

                    fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
                    fp.write(", ligand" + ", " + str(i) + "\n")
                    fp.write("load " + fp_path + 'pdbs_traj/test_p' + str(i) + '.pdb ')
                    fp.write(", protein" + ", " + str(i) + "\n")
