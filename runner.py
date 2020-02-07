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

checkpoint = "/Users/austin/checkpoint_101/checkpoint-101"

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
        # V-trace params (see vtrace.py).
        "vtrace": True,
        "vtrace_clip_rho_threshold": 1.0,
        "vtrace_clip_pg_rho_threshold": 1.0,

        # System params.
        #
        # == Overview of data flow in IMPALA ==
        # 1. Policy evaluation in parallel across `num_workers` actors produces
        #    batches of size `sample_batch_size * num_envs_per_worker`.
        # 2. If enabled, the replay buffer stores and produces batches of size
        #    `sample_batch_size * num_envs_per_worker`.
        # 3. If enabled, the minibatch ring buffer stores and replays batches of
        #    size `train_batch_size` up to `num_sgd_iter` times per batch.
        # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
        #    on batches of size `train_batch_size`.
        #
        "sample_batch_size": 16,
        "train_batch_size": 128,
        "min_iter_time_s": 10,
        "num_workers": 1,
        # number of GPUs the learner should use.
        "num_gpus": 0,
        # set >1 to load data into GPUs in parallel. Increases GPU memory usage
        # proportionally with the number of buffers.
        "num_data_loader_buffers": 1,
        # how many train batches should be retained for minibatching. This conf
        # only has an effect if `num_sgd_iter > 1`.
        "minibatch_buffer_size": 1,
        # number of passes to make over each train batch
        "num_sgd_iter": 2,
        # set >0 to enable experience replay. Saved samples will be replayed with
        # a p:1 proportion to new data samples.
        "replay_proportion": 0.0,
        # number of sample batches to store for replay. The number of transitions
        # saved total will be (replay_buffer_num_slots * sample_batch_size).
        "replay_buffer_num_slots": 0,
        # max queue size for train batches feeding into the learner
        "learner_queue_size": 16,
        # wait for train batches to be available in minibatch buffer queue
        # this many seconds. This may need to be increased e.g. when training
        # with a slow environment
        "learner_queue_timeout": 300,
        # level of queuing for sampling.
        "max_sample_requests_in_flight_per_worker": 2,
        # max number of workers to broadcast one set of weights to
        "broadcast_interval": 2,
        # use intermediate actors for multi-level aggregation. This can make sense
        # if ingesting >2GB/s of samples, or if the data requires decompression.
        "num_aggregation_workers": 1,

        # Learning params.
        "grad_clip": 1.0,
        # either "adam" or "rmsprop"
        "opt_type": "adam",
        "lr": 8e-5,
        "lr_schedule": None,
        # rmsprop considered
        "decay": 0.99,
        "momentum": 0.0,
        "epsilon": 0.1,
        # balancing the three losses
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "entropy_coeff_schedule": None,

        # use fake (infinite speed) sampler for testing
        "_fake_sampler": False,
        'env_config': envconf,
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 8,
        },
        'eager': False,
        'reuse_actors': False,
        "env": 'lactamase_docking',
        'log_level': "INFO",
        'vf_share_layers': True,
        'use_lstm': True
    }

    ppo_config = impala.DEFAULT_CONFIG
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
            for j in range(5):
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
