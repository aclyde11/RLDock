import copy
import glob
import math
from random import randint

import gym
import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation as R

from rldock.environments.LPDB import LigandPDB
from rldock.environments.utils import MultiScorerFromReceptor, Voxelizer, l2_action, \
    MinMax, ScorerFromReceptor, Scorer


class LactamaseDocking(gym.Env):
    metadata = {'render.modes': ['human']}

    ## Init the object
    def __init__(self, config, bypass=None):
        super(LactamaseDocking, self).__init__()
        self.config = config
        if bypass is not None:
            self.config.update(bypass)
            config.update(bypass)
        self.viewer = None
        self.logmessage(config)
        dims = np.array(config['bp_dimension']).flatten().astype(np.float32)
        self.logmessage("config after bypass", config)

        # # #
        # Used for bound checking
        # # #
        self.random_space_init = spaces.Box(low=-0.5 * dims,
                                            high=0.5 * dims,
                                            dtype=np.float32)
        self.random_space_rot = spaces.Box(low=0,
                                           high=2 * 3.1415926,
                                           dtype=np.float32,
                                           shape=(3, 1))

        # # #
        # Used for reset position if random set in envconf
        # # #
        self.random_space_init_reset = spaces.Box(low=-0.4 * dims,
                                                  high=0.4 * dims,
                                                  dtype=np.float32)
        self.random_space_rot_reset = spaces.Box(low=0,
                                                 high=2 * 3.1415926,
                                                 dtype=np.float32,
                                                 shape=(3, 1))

        self.voxelcache = {}
        self.use_random = True

        if config['discrete']:
            self.actions_multiplier = np.array(
                [(config['action_space_d'][i] / (config['K_trans'] - 1)) for i in range(3)]
                + [config['action_space_r'][i] / (config['K_theta'] - 1) for i in range(6)], dtype=np.float32)
            # self.action_space = spaces.MultiDiscrete([config['K_trans']] * 3 + [config['K_theta']] * 6)
            self.action_space = spaces.Tuple(
                [spaces.Discrete(config['K_trans'])] * 3 + [spaces.Discrete(config['K_theta'])] * 3)

        else:
            lows = -1 * np.array(list(config['action_space_d']) + list(config['action_space_r']), dtype=np.float32)
            highs = np.array(list(config['action_space_d']) + list(config['action_space_r']), dtype=np.float32)
            self.action_space = spaces.Box(low=lows,
                                           high=highs,
                                           dtype=np.float32)

        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=2, shape=config['output_size'], dtype=np.float32),
                                               spaces.Box(low=-np.inf, high=np.inf, shape=[2], dtype=np.float32)])

        self.voxelizer = Voxelizer(config['protein_wo_ligand'], config)
        if self.config['oe_box'] is None:
            self.oe_scorer = ScorerFromReceptor(
                self.make_receptor(self.config['protein_wo_ligand'], use_cache=config['use_cache_voxels']))
        else:
            self.logmessage("Found OE BOx for recetpor")
            self.oe_scorer = Scorer(config['oe_box'])

        # self.minmaxs = [MinMax(-278, -8.45), MinMax(-1.3, 306.15), MinMax(-17.52, 161.49), MinMax(-2, 25.3)]
        self.minmaxs = [MinMax()]

        self.reference_ligand = LigandPDB.parse(config['ligand'])
        self.reference_centers = self.reference_ligand.get_center()
        self.atom_center = LigandPDB.parse(config['ligand'])
        self.names = []

        if config['many_ligands'] and config['random_ligand_folder'] is not None:
            self.train_ligands()
        else:
            self.rligands = None

        self.cur_atom = copy.deepcopy(self.atom_center)
        self.trans = [0, 0, 0]
        self.rot = [0, 0, 0]
        self.rot = [0, 0, 0]
        self.steps = 0
        self.last_score = 0
        self.cur_reward_sum = 0
        self.name = ""

        self.receptor_refereence_file_name = config['protein_wo_ligand']

        self.ordered_recept_voxels = None
        # if self.config['movie_mode']:
        #     import os.path
        #     listings = glob.glob(self.config['protein_state_folder'] + "*.pdb")
        #     self.logmessage("listing len", len(listings))
        #     ordering = list(map(lambda x : int(str(os.path.basename(x)).split('.')[0].split("_")[-1]), listings))
        #     ordering = np.argsort(ordering)[:200]
        #     self.logmessage("Making ordering....")
        #     self.logmessage(listings[0], len(listings))
        #     self.ordered_recept_voxels = [listings[i] for i in ordering]

    def reset_ligand(self, newlig):
        """
        :param newlig: take a LPBD ligand and transform it to center reference
        :return: new LPDB
        """
        x, y, z = newlig.get_center()
        return newlig.translate(self.reference_centers[0] - x, self.reference_centers[1] - y,
                                self.reference_centers[2] - z)

    def align_rot(self):
        """
            Aligns interal rotations given angluar def in current version
        """
        for i in range(3):
            if self.rot[i] < 0:
                self.rot[i] = 2 * 3.14159265 + self.rot[i]
            self.rot[i] = self.rot[i] % (2 * 3.14159265)

    def get_action(self, action, use_decay=True):
        """
        Override this function if you want to modify the action in some deterministic sense...
        :param action: action from step funtion
        :return: action
        """
        if self.config['discrete']:
            action = np.array(action) * self.actions_multiplier
        else:
            action = np.array(action).flatten()

        if use_decay and self.config['decay'] is not None:
            action = action * math.pow(self.config['decay'], self.steps)
        return action

    def get_penalty_from_overlap(self, obs):
        """
        Evaluates from obs to create a reward value. Do not scale this value, save for weighting later in step function.
        :param obs: obs from model, or hidden env state
        :return: penalty for overlap, positive value
        """
        return np.sum(obs[0][:, :, :, -1] >= 2)

    def oe_score_combine(self, oescores, average=True):
        score = oescores[0]
        return score

    def get_rotation_matrix(self, rot):
        return R.from_euler('xyz', rot, degrees=False).as_matrix()

    def get_rotation(self, rot):
        return rot

    def step(self, action):
        if np.any(np.isnan(action)):
            self.logerror("ERROR, nan action from get action", action)

        action = self.get_action(action)
        rotM = self.get_rotation_matrix(action[3:])
        assert (action.shape[0] == 6)

        self.cur_atom = self.translate_molecule(self.cur_atom, action[0], action[1], action[2])
        self.cur_atom = self.rotate_molecule(self.cur_atom, rotM)

        self.trans[0] += action[0]
        self.trans[1] += action[1]
        self.trans[2] += action[2]
        self.rot = np.matmul(self.rot, rotM)
        self.steps += 1

        oe_score = self.oe_scorer(self.cur_atom.toPDB())
        oe_score = self.oe_score_combine(oe_score)
        reset = self.decide_reset(oe_score)
        improve = oe_score - self.last_score
        self.last_score = oe_score
        obs = self.get_obs()

        w1 = min(0, -1 * self.config['improve_weight'] * improve)
        w2 = -1 * self.config['l2_decay'] * l2_action(action, self.steps)
        w3 = -1 * self.config['overlap_weight'] * self.get_penalty_from_overlap(obs)
        w4 = -1 * self.config['score_weight'] * oe_score
        reward = w1 + w4 + w2 * + w3

        if self.config['reward_ramp'] is not None:
            ramp = max(1.0, self.config['reward_ramp'] * min(1.0, ((self.steps * self.steps - 35) / 20)))
            reward = reward * ramp
        else:
            ramp = None

        if reset:
            reward = oe_score * -1.0
            self.logmessage("final reset value, replaces reward", reward)

        self.logmessage(
            {"reward": reward,
             "ramp": ramp,
             "w1": w1,
             "w2": w2,
             "w3": w3,
             "w4": w4,
             'cur_step': self.steps,
             'oe_score': oe_score,
             'trans': self.trans,
             'reset': reset})

        self.last_reward = reward
        self.cur_reward_sum += reward

        if self.config['movie_mode']:
            self.movie_step(self.steps)

        assert (not np.any(np.isnan(reward)))
        assert (not np.any(np.isnan(obs[0])))
        assert (not np.any(np.isnan(obs[1])))

        return obs, \
               reward, \
               reset, \
               {'atom': self.cur_atom.toPDB(),
                'protein': self.receptor_refereence_file_name}

    def decide_reset(self, score):
        return (self.steps >= self.config['max_steps'])

    def get_state_vector(self):
        max_steps = self.steps / self.config['max_steps']
        return np.nan_to_num(np.array([float(np.clip(self.last_score, -30, 30)), max_steps]).astype(np.float32),
                             posinf=100, neginf=-100, nan=-100)

    def logmessage(self, *args, **kwargs):
        if self.config['debug']:
            print(*args, **kwargs)

    def logerror(self, *args, **kwargs):
        print(*args, **kwargs)
        exit()

    def reset_random_recep(self):
        import random as rs
        self.receptor_refereence_file_name = list(self.voxelcache.keys())[rs.randint(0, len(self.voxelcache) - 1)]
        self.voxelizer, self.oe_scorer = self.voxelcache[self.receptor_refereence_file_name]

    def movie_step(self, step=0):
        try:
            self.receptor_refereence_file_name = self.ordered_recept_voxels[step]
        except:
            self.logerror("Error length...", len(self.ordered_recept_voxels))
            exit()

        pdb_file_name = self.receptor_refereence_file_name

        if pdb_file_name in self.voxelcache:
            self.voxelizer, self.oe_scorer = self.voxelcache[pdb_file_name]
        else:
            try:
                self.logmessage("Not in cache, making....", pdb_file_name)
                self.voxelizer = Voxelizer(pdb_file_name, self.config, write_cache=True)
                recept = self.make_receptor(pdb_file_name)
                self.oe_scorer = MultiScorerFromReceptor(recept)
                self.voxelcache[pdb_file_name] = (self.voxelizer, self.oe_scorer)
            except:
                self.logerror("Error, not change.")

    def reset(self, random=None, many_ligands=None, random_dcd=None, load_num=None):
        random = random or self.config['random']
        many_ligands = many_ligands or self.config['many_ligands']
        random_dcd = random_dcd or self.config['random_dcd']
        load_num = load_num or self.config['load_num']

        # if self.config['movie_mode']:
        #     import random as rs
        #     self.movie_step(rs.randint(0, 50))
        # elif random_dcd:
        #     import random as rs
        #     if len(self.voxelcache) < load_num:
        #         self.logmessage("Voxel cache is empty or with size", len(self.voxelcache))
        #         listings = glob.glob(self.config['protein_state_folder'] + "*.pdb")
        #         self.logmessage("Found", len(listings), "protein states in folder.")
        #         while len(self.voxelcache) < load_num:
        #             pdb_file_name = rs.choice(listings)
        #
        #             if pdb_file_name in self.voxelcache:
        #                 self.voxelizer, self.oe_scorer = self.voxelcache[pdb_file_name]
        #             else:
        #                 try:
        #                     self.logmessage("Not in cache, making....", pdb_file_name)
        #                     self.voxelizer = Voxelizer(pdb_file_name, self.config, write_cache=True)
        #                     recept = self.make_receptor(pdb_file_name)
        #                     self.oe_scorer = MultiScorerFromReceptor(recept)
        #                     self.voxelcache[pdb_file_name] = (self.voxelizer, self.oe_scorer)
        #                 except:
        #                     self.logerror("Error, not change.")
        #
        #     self.reset_random_recep()

        if many_ligands and self.rligands != None and self.use_random:
            idz = randint(0, len(self.rligands) - 1)
            start_atom = copy.deepcopy(self.rligands[idz])
            self.name = self.names[idz]
        elif many_ligands and self.rligands != None:
            start_atom = copy.deepcopy(self.rligands.pop(0))
            self.name = self.names.pop(0)
        else:
            start_atom = copy.deepcopy(self.atom_center)

        if random is not None and float(random) != 0:
            x, y, z, = self.random_space_init_reset.sample().flatten().ravel() * float(random)
            x_theta, y_theta, z_theta = self.random_space_rot_reset.sample().flatten().ravel() * float(random)
            rot = self.get_rotation_matrix(np.array([x_theta, y_theta, z_theta]))
            self.trans = [x, y, z]
            self.rot = rot
            random_pos = self.translate_molecule(start_atom, x, y, z)
            random_pos = self.rotate_molecule(random_pos, rot)
        else:
            if self.config['ref_ligand_move'] is not None:
                self.trans = self.config['ref_ligand_move']
            else:
                self.trans = [0, 0, 0]
            random_pos = self.translate_molecule(start_atom, *self.trans)

        self.cur_atom = random_pos
        self.last_score = self.oe_score_combine(self.oe_scorer(self.cur_atom.toPDB()))
        self.steps = 0
        self.cur_reward_sum = 0
        self.last_reward = 0
        self.next_exit = False
        self.decay_v = 1.0

        self.logmessage("Reset ligand", self.trans, self.rot)
        return self.get_obs()

    def get_obs(self, quantity='all'):
        x = self.voxelizer(self.cur_atom.toPDB(), quantity=quantity).squeeze(0).astype(np.float32)
        oe_score = self.oe_scorer(self.cur_atom.toPDB())
        oe_score = self.oe_score_combine(oe_score)
        return (x, np.array([0.0, self.steps]))

    def make_receptor(self, pdb, use_cache=True):
        from openeye import oedocking, oechem
        import os.path

        file_name = str(os.path.basename(pdb))
        check_oeb = self.config['cache'] + file_name.split(".")[0] + ".oeb"
        if use_cache and os.path.isfile(check_oeb):
            self.logmessage("Using stored receptor", check_oeb)

            g = oechem.OEGraphMol()
            oedocking.OEReadReceptorFile(g, check_oeb)
            return g
        else:
            self.logmessage("NO OEBOX, creating recetpor on fly for base protein", check_oeb, pdb)

            proteinStructure = oechem.OEGraphMol()
            ifs = oechem.oemolistream(pdb)
            ifs.SetFormat(oechem.OEFormat_PDB)
            oechem.OEReadMolecule(ifs, proteinStructure)

            box = oedocking.OEBox(*self.config['bp_max'], *self.config['bp_min'])

            receptor = oechem.OEGraphMol()
            s = oedocking.OEMakeReceptor(receptor, proteinStructure, box)
            self.logmessage("make_receptor bool", s)
            oedocking.OEWriteReceptorFile(receptor, check_oeb)
            return receptor

    def render(self, mode='human'):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib import pyplot as plt

        obs = (self.get_obs(quantity='ligand')[:, :, :, -1]).squeeze()
        obs1 = (self.get_obs(quantity='protein')[:, :, :, -1]).squeeze()
        # np.save("/Users/austin/obs.npy", obs)
        # np.save("/Users/austin/pro.npy", obs1)
        print(obs.shape)
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.gca(projection='3d')
        canvas = FigureCanvas(fig)

        coords_x = []
        coords_y = []
        coords_z = []
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                for z in range(obs.shape[2]):
                    if obs[i, j, z] == 1:
                        coords_x.append(i)
                        coords_y.append(j)
                        coords_z.append(z)

        coords_x1 = []
        coords_y1 = []
        coords_z1 = []
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                for z in range(obs.shape[2]):
                    if obs1[i, j, z] == 1:
                        coords_x1.append(i)
                        coords_y1.append(j)
                        coords_z1.append(z)

        ax.set_title("Current step:" + str(self.steps) + ", Curr Reward" + str(self.last_reward) + ', Curr RSUm' + str(
            self.cur_reward_sum) + 'score' + str(self.last_score))
        try:
            ax.plot_trisurf(coords_x, coords_y, coords_z, linewidth=0.2, antialiased=True)
            ax.plot_trisurf(coords_x1, coords_y1, coords_z1, linewidth=0.2, antialiased=True, alpha=0.5)

        except:
            pass

        ax.set_xlim(0, 40)
        ax.set_ylim(0, 40)
        ax.set_zlim(0, 40)
        # fig.show()
        canvas.draw()  # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(100 * 10, 100 * 10, 3)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        pass

    def check_atom_in_box(self):
        return self.random_space_init.contains(self.trans)

    def disable_random(self):
        self.use_random = False

    def eval_ligands(self):
        self.rligands = glob.glob(self.config['random_ligand_folder_test'] + "/*.pdb")
        self.names = copy.deepcopy(self.rligands)
        self.names = list(map(lambda x: x.split('/')[-1].split('.')[0], self.rligands))

        for i in range(len(self.rligands)):
            self.rligands[i] = self.reset_ligand(LigandPDB.parse(self.rligands[i]))

    def train_ligands(self):
        self.rligands = glob.glob(self.config['random_ligand_folder'] + "/*.pdb") + [self.config['ligand']]
        self.names = list(map(lambda x: x.split('/')[-1].split('.')[0], self.rligands))

        for i in range(len(self.rligands)):
            self.rligands[i] = self.reset_ligand(LigandPDB.parse(self.rligands[i]))
        assert (len(self.rligands) == len(self.names))

    def rotate_molecule(self, mol, *args, **kwargs):
        if not self.config['action_space_r_stop']:
            return mol.rotateM(*args, **kwargs)
        else:
            return mol

    def translate_molecule(self, mol, *args, **kwargs):
        if not self.config['action_space_d_stop']:
            return mol.translate(*args, **kwargs)
        else:
            return mol
