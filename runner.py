import argparse
import shutil
import ray
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.impala import impala
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

checkpoint = "/Users/austin/checkpoint_11/checkpoint-11"

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


    d = {
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 16,
        },
        'gamma' : 0.9,
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": False,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        'use_lstm': True,
        # The GAE(lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.5,
        # Size of batches collected from each worker.
        "sample_batch_size": 50,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 250,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 100,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": False,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 10,
        # Stepsize of SGD.
        "lr": 8e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": True,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1e-2,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.01,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.2,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": 40.0,
        # Target value for KL divergence.
        "kl_target": 0.01,
        'env_config' : envconf,
        "num_gpus": 0,
        "num_workers" :1,
        'batch_mode' : 'complete_episodes',
        'horizon' : 50
    }

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
            for j in range(1):
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
