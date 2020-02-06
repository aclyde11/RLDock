from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rldock.environments.lactamase import LactamaseDocking
import gym
from gym.spaces import Discrete
import numpy as np
import random
import argparse
from config import config as envconf
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.impala import impala
from ray.tune.logger import pretty_print

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()



class MyKerasRNN(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=128,
                 cell_size=64):
        super(MyKerasRNN, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        self.cell_size = cell_size
        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, envconf['output_size'][0], envconf['output_size'][1], envconf['output_size'][2], envconf['output_size'][3]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        h = tf.keras.layers.Reshape([-1] + list(envconf['output_size']))(input_layer)

        h = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv3D(filters=32, kernel_size=6, padding='valid', name='notconv1'))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(64, 6, padding='valid', name='conv3d_2'))(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                                                         strides=None,
                                                                         padding='valid'))(h)
        h = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding='valid', name='notconv12'))(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_22'))(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)

        h = tf.keras.layers.Reshape([-1, 11 * 11 * 11 * 32])(h)
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(h)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
            inputs=dense1,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])



def env_creator(env_config):
    return LactamaseDocking(env_config)
    # return an env instance

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--ncpu', type=int, default=4)
    parser.add_argument('--local', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    register_env("lactamase_docking", env_creator)
    args = get_args()
    if args.local:
        ray.init()
    else:
        memory_story = 256.00  * 1e+9
        obj_store = 128.00 * 1e+9
        ray.init(memory=memory_story, object_store_memory=obj_store)
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
        "num_workers": args.ncpu,
        # number of GPUs the learner should use.
        "num_gpus": args.ngpu,
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
        'env_config' : envconf,
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 8,
        },
        'eager': False,
        'reuse_actors': False,
        "env": 'lactamase_docking',
        'log_level': "INFO",
        'vf_share_layers': True,
        'use_lstm' : True
    }

    ppo_config = impala.DEFAULT_CONFIG
    ppo_config.update(d)

    trainer = impala.ImpalaTrainer(config=ppo_config, env='lactamase_docking')


    # d = {
    #     "model": {
    #         "custom_model": "rnn",
    #         "max_seq_len": 8,
    #     },
    #     # Should use a critic as a baseline (otherwise don't use value baseline;
    #     # required for using GAE).
    #     "use_critic": True,
    #     # If true, use the Generalized Advantage Estimator (GAE)
    #     # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    #     "use_gae": True,
    #     'use_lstm': True,
    #     # The GAE(lambda) parameter.
    #     "lambda": 1.0,
    #     # Initial coefficient for KL divergence.
    #     "kl_coeff": 0.2,
    #     # Size of batches collected from each worker.
    #     "sample_batch_size": 64,
    #     # Number of timesteps collected for each SGD round. This defines the size
    #     # of each SGD epoch.
    #     "train_batch_size": 512,
    #     # Total SGD batch size across all devices for SGD. This defines the
    #     # minibatch size within each epoch.
    #     "sgd_minibatch_size": 64,
    #     # Whether to shuffle sequences in the batch when training (recommended).
    #     "shuffle_sequences": False,
    #     # Number of SGD iterations in each outer loop (i.e., number of epochs to
    #     # execute per train batch).
    #     "num_sgd_iter": 30,
    #     # Stepsize of SGD.
    #     "lr": 5e-5,
    #     # Learning rate schedule.
    #     "lr_schedule": None,
    #     # Share layers for value function. If you set this to True, it's important
    #     # to tune vf_loss_coeff.
    #     "vf_share_layers": True,
    #     # Coefficient of the value function loss. IMPORTANT: you must tune this if
    #     # you set vf_share_layers: True.
    #     "vf_loss_coeff": 1.0,
    #     # Coefficient of the entropy regularizer.
    #     "entropy_coeff": 0.0,
    #     # Decay schedule for the entropy regularizer.
    #     "entropy_coeff_schedule": None,
    #     # PPO clip parameter.
    #     "clip_param": 0.3,
    #     # Clip param for the value function. Note that this is sensitive to the
    #     # scale of the rewards. If your expected V is large, increase this.
    #     "vf_clip_param": 10.0,
    #     # If specified, clip the global norm of gradients by this amount.
    #     "grad_clip": 10.0,
    #     # Target value for KL divergence.
    #     "kl_target": 0.01,
    #     'env_config' : envconf,
    #     "num_gpus": args.ngpu,
    #     "num_workers" : args.ncpu
    # }
    # ppo_config = ppo.DEFAULT_CONFIG
    # ppo_config.update(d)
    # trainer = ppo.PPOTrainer(config=ppo_config, env='lactamase_docking')


    for i in range(1000):
        result = trainer.train()

        if i % 1 == 0:
            print(pretty_print(result))

        if i % 25 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
