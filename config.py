import os
import math

path = os.path.dirname(os.path.abspath(__file__)) + "/resources"
# path = ""
config = {
    'discrete': False,
    'K_trans': 4,
    'K_theta': 4,
    'normalize': True,
    'action_space_d': (0.25, 0.25, 0.25),
    'action_space_r': (math.pi * 2 * 0.05, math.pi * 2 * 0.05, math.pi * 2 * 0.05),
    'action_space_r_stop': True,
    'action_space_d_stop': False,
    'protein_wo_ligand': path + '/6dpz/6pdz_wo_ligand.pdb',
    'ligand': path + '/6dpz/6dpz_ligand.pdb',
    'oe_box': path + "/6dpz/oe_box_6dpz.oeb",
    'bp_dimension': [16, 16, 16],
    'bp_centers': [30.748, 0.328, 23.421],
    'bp_min': [22.748, -7.672, 13.421],
    'bp_max': [38.748, 8.328, 31.421],
    'voxelsize': 0.33,
    'output_size': (49, 49, 49, 8),  # (39,40,42,8),
    'max_steps': 50,
    'decay': 0.98,  # ^25 = 0.001,
    'voxel_method': 'C',
    'debug': False,
    'reward_ramp': 1.0,

    ## Reward function tuning
    'overlap_weight': 0.000001,
    'l2_decay': 0.000001,
    'improve_weight': 0.1,
    'score_weight': 0.01,

    ## Ligand and Protein selection features
    'random': 0.0,  # randomly place ligand around protein
    'many_ligands': False,  # use many ligands from the random_ligand_folder
    'random_ligand_folder': path + '/rligands',
    'random_ligand_folder_test': path + '/rligands_eval',
    # use train_ligands() or eval_ligands() to switch, default train_ligands called if manyligands not false
    'random_dcd': False,  # use random protein states from folder
    'protein_state_folder': '/Users/austin/gpcr/structs/',  # *.pdbs used randomly
    'load_num': 3,  # used for speed, set number of states each env uses for training.
    'cache': 'cache/',
    'use_cache_voxels': False,

    'ref_ligand_move': None,  # move GPCR ligand out of reference pocket
    'movie_mode': False
}
config['bp_min'] = [config['bp_centers'][i] - int(config['bp_dimension'][i] / 2) for i in range(3)]
config['bp_max'] = [config['bp_centers'][i] + int(config['bp_dimension'][i] / 2) for i in range(3)]
