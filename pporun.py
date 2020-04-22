import faulthandler
import sys

faulthandler.enable(file=sys.stderr, all_threads=True)
import ray
from ray.rllib.agents.impala import impala
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo import appo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from argparse import ArgumentParser
from ray import tune

from config import config as envconf
from ray.tune.registry import register_env

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from rldock.voxel_policy.utils_tf2 import lrelu
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
from rldock.environments.lactamase import  LactamaseDocking
from resnet import Resnet3DBuilder
tf = try_import_tf()

class DeepDrug3D(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(DeepDrug3D, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)

        print(obs_space)
        self.inputs = [tf.keras.layers.Input(
            shape=(26, 27, 28, 8), name="observations"),


        tf.keras.layers.Input(shape=(8,), name='state_vec_obs')]
        h = tf.keras.layers.Conv3D(filters=32,  kernel_size=5, padding='valid', name='notconv1')(self.inputs[0])
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_2')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
            strides=None,
            padding='valid')(h)
        h = tf.keras.layers.Conv3D(filters=32,  kernel_size=5, padding='valid', name='notconv12')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_22')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
            strides=None,
            padding='valid')(h)

        h = tf.keras.layers.Flatten()(h)

        layer_2 = tf.keras.layers.Dense(64, activation=lrelu)(self.inputs[1])
        layer_2 = tf.keras.layers.Concatenate()([layer_2, h])

        layer_2 = tf.keras.layers.Dense(64, activation=lrelu)(layer_2)
        layer_2 = tf.keras.layers.Dense(256, activation=lrelu)(layer_2)


        layer_4p = tf.keras.layers.Dense(256, activation='relu', name='ftp2')(layer_2)
        layer_5p = tf.keras.layers.Dense(64, activation=lrelu, name='ftp3')(layer_4p)

        layer_4v = tf.keras.layers.Dense(256, activation='relu', name='ftv2')(layer_2)
        layer_5v = tf.keras.layers.Dense(64, activation=lrelu, name='ftv3')(layer_4v)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation='sigmoid',
            kernel_initializer=normc_initializer(0.2))(layer_5p)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.1))(layer_5v)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):

        model_out, self._value_out = self.base_model([input_dict["obs"]['image'], input_dict['obs']['state_vector']])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])



class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")

        layer_14 =  Resnet3DBuilder.build_resnet_34(self.inputs, 400)


        layer_2 = tf.keras.layers.Flatten()(layer_14)
        layer_3p = tf.keras.layers.Dense(256, activation='relu', name='ftp')(layer_2)
        layer_4p = tf.keras.layers.Dense(128, activation='relu', name='ftp2')(layer_3p)
        layer_5p = tf.keras.layers.Dense(64, activation=lrelu, name='ftp3')(layer_4p)

        layer_3v = tf.keras.layers.Dense(256, activation='relu', name='ftv')(layer_2)
        layer_4v = tf.keras.layers.Dense(128, activation='relu', name='ftv2')(layer_3v)
        layer_5v = tf.keras.layers.Dense(64, activation=lrelu, name='ftv3')(layer_4v)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation='hard_sigmoid',
            kernel_initializer=normc_initializer(0.1))(layer_5p)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.1))(layer_5v)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

        self.base_model.load_weights('my_model_weights.h5')

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# memory_story = 256.00  * 1e+9
# obj_store = 128.00 * 1e+9
# ray.init(memory=memory_story, object_store_memory=obj_store)
ray.init()

parser = ArgumentParser()
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--ncpu', type=int, default=4)
args = parser.parse_args()

ModelCatalog.register_custom_model("keras_model", MyKerasModel)
ModelCatalog.register_custom_model("deepdrug3d", DeepDrug3D)

def env_creator(env_config):
    return LactamaseDocking(env_config)  # return an env instance

register_env("lactamase_docking", env_creator)

config = ppo.DEFAULT_CONFIG.copy()
config['log_level'] = 'INFO'

ppo_conf = {"lambda": 0.95,
            "kl_coeff": 1.0,
            "sgd_minibatch_size": 64,
            "shuffle_sequences": True,
            "num_sgd_iter": 10,
            "lr": 1e-4,
            "lr_schedule": None,
            "vf_share_layers": False,
            "vf_loss_coeff": 0.75,
            "entropy_coeff": 0.001,
            "entropy_coeff_schedule": None,
            "clip_param": 0.2,
            "kl_target": 0.01,
            "gamma" : 0.99
        }


config.update(ppo_conf)

config['sample_batch_size'] = 32
config['train_batch_size'] = 256

config["num_gpus"] = args.ngpu  # used for trainer process
config["num_workers"] = args.ncpu
config['num_envs_per_worker'] = 1
config['env_config'] = envconf
config['model'] = {"custom_model": 'deepdrug3d'}
config['horizon'] = envconf['max_steps']

# trainer = impala.ImpalaTrainer(config=config, env='lactamase_docking')
trainer = ppo.PPOTrainer(config=config, env="lactamase_docking")
# trainer.restore('/homes/aclyde11/ray_results/PPO_lactamase_docking_2019-11-18_13-40-14ihwtk2lw/checkpoint_51/checkpoint-51')
policy = trainer.get_policy()
print(policy.model.base_model.summary())

config['env'] = 'lactamase_docking'

for i in range(1000):
    result = trainer.train()

    if i % 1 == 0:
        print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
