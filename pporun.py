import faulthandler
import sys

from ray.rllib.utils.annotations import override
import tensorflow_probability as tfp

faulthandler.enable(file=sys.stderr, all_threads=True)
import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.agents.impala import impala
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo import appo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from argparse import ArgumentParser
from ray.tune import register_env
from ray import tune
import numpy as np
from config import config as envconf
from ray.tune.registry import register_env
from ray.rllib.models.tf.tf_action_dist import Deterministic, TFActionDistribution, ActionDistribution
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from rldock.voxel_policy.utils_tf2 import lrelu
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
from rldock.environments.lactamase import LactamaseDocking
from resnet import Resnet3DBuilder

tf = try_import_tf()


class Ordinal(TFActionDistribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, inputs, model=None):
        self.inputs = inputs
        L = tf.sigmoid(self.inputs)
        l_minus = tf.log(1 - tf.identity(L))
        L = tf.log(L)

        ascend = tf.dtypes.cast(tf.expand_dims(tf.range(1, L.shape[-1] + 1), 0), tf.float32)
        desend = tf.dtypes.cast(tf.expand_dims(tf.reverse(tf.range(L.shape[-1]), axis=[0]), 0), tf.float32)

        L_prime = ascend * L + desend * l_minus
        self.inputs = tf.nn.softmax(L_prime, axis=-1)

        super(Ordinal, self).__init__(self.inputs, model)

    @override(ActionDistribution)
    def logp(self, x):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.inputs, labels=tf.cast(x, tf.int32))

    @override(ActionDistribution)
    def entropy(self):
        a0 = self.inputs - tf.reduce_max(
            self.inputs, reduction_indices=[1], keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, reduction_indices=[1], keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), reduction_indices=[1])

    @override(ActionDistribution)
    def kl(self, other):
        a0 = self.inputs - tf.reduce_max(
            self.inputs, reduction_indices=[1], keep_dims=True)
        a1 = other.inputs - tf.reduce_max(
            other.inputs, reduction_indices=[1], keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, reduction_indices=[1], keep_dims=True)
        z1 = tf.reduce_sum(ea1, reduction_indices=[1], keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(
            p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), reduction_indices=[1])

    @override(TFActionDistribution)
    def _build_sample_op(self):
        t = tf.squeeze(tf.multinomial(self.inputs, 1), axis=1)
        t = tf.clip_by_value(t, clip_value_min=0, clip_value_max=envconf['K_trans'] - 1)
        return t

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return action_space.n


class MultiOrdinal(TFActionDistribution):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    def __init__(self, inputs, model, input_lens=([envconf['K_trans']] * 3 + [envconf['K_theta']] * 6)):
        # skip TFActionDistribution init
        ActionDistribution.__init__(self, inputs, model)
        self.cats = [
            Ordinal(input_, model)
            for input_ in tf.split(inputs, input_lens, axis=1)
        ]
        self.sample_op = self._build_sample_op()

    @override(ActionDistribution)
    def logp(self, actions):
        # If tensor is provided, unstack it into list
        if isinstance(actions, tf.Tensor):
            actions = tf.unstack(tf.cast(actions, tf.int32), axis=1)
        logps = tf.stack(
            [cat.logp(act) for cat, act in zip(self.cats, actions)])
        return tf.math.reduce_sum(logps, axis=0)

    @override(ActionDistribution)
    def multi_entropy(self):
        return tf.stack([cat.entropy() for cat in self.cats], axis=1)

    @override(ActionDistribution)
    def entropy(self):
        return tf.math.reduce_sum(self.multi_entropy(), axis=1)

    @override(ActionDistribution)
    def multi_kl(self, other):

        ress = []
        for cat, oth_cat in zip(self.cats, other.cats):
            res = tf.expand_dims(cat.kl(oth_cat), 1)
            ress.append(res)
        return ress

    @override(ActionDistribution)
    def kl(self, other):
        kls = tf.concat(self.multi_kl(other), axis=-1)
        t = tf.math.reduce_sum(kls, axis=-1, keepdims=True)
        return t

    @override(TFActionDistribution)
    def _build_sample_op(self):
        return tf.stack([cat.sample() for cat in self.cats], axis=1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.sum(action_space.nvec)


class MyActionDist(TFActionDistribution):
    def __init__(self, inputs, model):
        mean, log_std = tf.split(inputs, 2, axis=1)
        self.mean = mean
        self.std = log_std
        self.dist = tfp.distributions.Beta(self.mean, self.std, validate_args=False, allow_nan_stats=True)
        TFActionDistribution.__init__(self, inputs, model)

    @override(ActionDistribution)
    def logp(self, x):
        t = tf.reduce_sum(self.dist.log_prob(x), reduction_indices=[1])
        return t

    @override(ActionDistribution)
    def kl(self, other):
        assert isinstance(other, MyActionDist)
        t = tf.reduce_sum(self.dist.kl_divergence(other.dist), reduction_indices=[1])
        return t

    @override(ActionDistribution)
    def entropy(self):
        t = tf.reduce_sum(self.dist.entropy(), reduction_indices=[1])
        return t

    @override(TFActionDistribution)
    def _build_sample_op(self):
        x = self.dist.sample()
        return x

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape) * 2

def __init__(self, *args, **kwargs):
    super(MyModelClass, self).__init__(*args, **kwargs)
    cell_size = 256

    # Define input layers
    input_layer = tf.keras.layers.Input(
        shape=(None, obs_space.shape[0]))
    state_in_h = tf.keras.layers.Input(shape=(256, ))
    state_in_c = tf.keras.layers.Input(shape=(256, ))
    seq_in = tf.keras.layers.Input(shape=(), dtype=tf.int32)

    # Send to LSTM cell
    lstm_out, state_h, state_c = tf.keras.layers.LSTM(
        cell_size, return_sequences=True, return_state=True,
        name="lstm")(
            inputs=input_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])
    output_layer = tf.keras.layers.Dense(...)(lstm_out)

    # Create the RNN model
    self.rnn_model = tf.keras.Model(
        inputs=[input_layer, seq_in, state_in_h, state_in_c],
        outputs=[output_layer, state_h, state_c])
    self.register_variables(self.rnn_model.variables)
    self.rnn_model.summary()
class MyKerasRNN(RecurrentTFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256):
        super(MyKerasRNN, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)

        # Define input layers
        state_in_h = tf.keras.layers.Input(shape=(hiddens_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(hiddens_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        self.inputs = tf.keras.layers.Input(
            shape=(26, 27, 28, 8), name="observations")
        h = tf.keras.layers.Conv3D(filters=32, kernel_size=5, padding='valid', name='notconv1')(self.inputs)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(64, 3, padding='valid', name='conv3d_2')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                         strides=None,
                                         padding='valid')(h)
        h = tf.keras.layers.Conv3D(filters=32, kernel_size=5, padding='valid', name='notconv12')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_22')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                         strides=None,
                                         padding='valid')(h)

        h = tf.keras.layers.Flatten()(h)

        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(h)

        print(dense1.shape, seq_in.shape, state_in_c.shape, state_in_h.shape)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            hiddens_size, return_sequences=True, return_state=True, name="lstm")(
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
            inputs=[self.input, seq_in, state_in_h, state_in_c],
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


class DeepDrug3D(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(DeepDrug3D, self).__init__(obs_space, action_space,
                                         num_outputs, model_config, name)

        print(obs_space)
        self.inputs = [tf.keras.layers.Input(
            shape=(26, 27, 28, 8), name="observations"), tf.keras.layers.Input(shape=(2,), name='state_vec_obs')]
        h = tf.keras.layers.Conv3D(filters=32, kernel_size=5, padding='valid', name='notconv1')(self.inputs[0])
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(64, 3, padding='valid', name='conv3d_2')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                         strides=None,
                                         padding='valid')(h)
        h = tf.keras.layers.Conv3D(filters=32, kernel_size=5, padding='valid', name='notconv12')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_22')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                         strides=None,
                                         padding='valid')(h)

        h = tf.keras.layers.Flatten()(h)

        layer_2 = tf.keras.layers.Concatenate()([self.inputs[1], h])
        layer_2 = tf.keras.layers.Dense(64, activation=lrelu)(layer_2)

        layer_2 = tf.keras.layers.Dense(64, activation=lrelu)(layer_2)
        layer_2 = tf.keras.layers.BatchNormalization()(layer_2)
        layer_2 = tf.keras.layers.Dense(128, activation=lrelu)(layer_2)
        layer_2 = tf.keras.layers.BatchNormalization()(layer_2)

        layer_4p = tf.keras.layers.Dense(128, activation='relu', name='ftp2')(layer_2)
        layer_4p = tf.keras.layers.BatchNormalization()(layer_4p)
        layer_5p = tf.keras.layers.Dense(64, activation=lrelu, name='ftp3')(layer_4p)

        layer_4v = tf.keras.layers.Dense(128, activation='relu', name='ftv2')(layer_2)
        layer_4v = tf.keras.layers.BatchNormalization()(layer_4v)
        layer_5v = tf.keras.layers.Dense(64, activation=lrelu, name='ftv3')(layer_4v)
        clipped_relu = lambda x: tf.clip_by_value(x, clip_value_min=1, clip_value_max=100)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.25))(layer_5p)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.1))(layer_5v)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model([
            tf.dtypes.cast(
                input_dict["obs"]['image'],
                tf.float32
            ),
            tf.dtypes.cast(input_dict['obs']['state_vector'], tf.float32)])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def env_creator(config):
    return LactamaseDocking(config)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--ncpu', type=int, default=4)
    parser.add_argument('--local', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.local:
        ray.init()
    else:
        memory_story = 256.00 * 1e+9
        obj_store = 128.00 * 1e+9
        ray.init(memory=memory_story, object_store_memory=obj_store)

    ModelCatalog.register_custom_model("deepdrug3d", DeepDrug3D)
    ModelCatalog.register_custom_model("rnn", MyKerasRNN)
    register_env("lactamase_docking", env_creator)

    config = ppo.DEFAULT_CONFIG.copy()
    ppo_conf = {"lambda": 1.0,
                "kl_coeff": 0.2,
                "sgd_minibatch_size": 32,
                "shuffle_sequences": False,
                "num_sgd_iter": 200,
                "train_batch_size": 4000,
                "lr": 5e-5,
                "vf_share_layers": True,
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "clip_param": 0.3,
                "vf_clip_param": 10.0,
                "grad_clip": 1.0,
                "kl_target": 0.01,
                "gamma": 0.9,
                'eager': True
                }
    config.update(ppo_conf)
    config['log_level'] = 'INFO'

    config["num_gpus"] = args.ngpu
    config["num_workers"] = args.ncpu
    config['num_envs_per_worker'] = 1
    config['model'] = {
        "custom_model": "rnn",
        "max_seq_len": 16,
    }
    config['env_config'] = envconf
    config['eager'] = True

    trainer = ppo.PPOTrainer(config=config, env='lactamase_docking')
    policy = trainer.get_policy()
    print(policy.model.base_model.summary())

    config['env'] = 'lactamase_docking'

    for i in range(250):
        result = trainer.train()

        if i % 1 == 0:
            print(pretty_print(result))

        if i % 25 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

