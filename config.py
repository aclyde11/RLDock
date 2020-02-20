import os
import math

path = os.path.dirname(os.path.abspath(__file__)) + "/resources"
# path = ""

bp_centers = [  30.784,   7.100,  31.738]
bp_dim = [16, 16, 16]
config = {
    'discrete': False,
    'K_trans': 4,
    'K_theta': 4,
    'normalize': True,
    'action_space_d': (0.33, 0.33, 0.33),
    'action_space_r': (math.pi * 2 * 0.05, math.pi * 2 * 0.05, math.pi * 2 * 0.05),
    'action_space_r_stop': True,
    'action_space_d_stop': False,
    'protein_wo_ligand': path + '/6dpz/6pdz_wo_ligand.pdb',
    'ligand': path + '/6dpz/6dpz_ligand.pdb',
    'oe_box': None, # path + "/6dpz/oe_box_6dpz.oeb",
    'bp_dimension': bp_dim,
    'bp_centers': bp_centers,
    'bp_min': [bp_centers[i] - int(bp_dim[i] / 2) for i in range(3)],
    'bp_max': [bp_centers[i] + int(bp_dim[i] / 2) for i in range(3)],
    'voxelsize': 0.33,
    'output_size': (49, 49, 49, 8),  # (39,40,42,8),
    'max_steps': 50,
    'decay': 0.97,  # ^25 = 0.001,
    'voxel_method': 'C',
    'debug': False,
    'reward_ramp': 1.0,

    ## Reward function tuning
    'overlap_weight': 0.0001,
    'l2_decay': 0.0001,
    'improve_weight': 1,
    'score_weight': 0.01,
    'oe_score' : "Chemgauss4",

    ## Ligand and Protein selection features
    'random': 0.3,  # randomly place ligand around protein
    'many_ligands': False,  # use many ligands from the random_ligand_folder
    'random_ligand_folder': path + '/rligands',
    'random_ligand_folder_test': path + '/rligands_eval',
    # use train_ligands() or eval_ligands() to switch, default train_ligands called if manyligands not false
    'random_dcd': False,  # use random protein states from folder
    'protein_state_folder': '/Users/austin/gpcr/structs/',  # *.pdbs used randomly
    'load_num': 3,  # used for speed, set number of states each env uses for training.
    'cache': 'cache/',
    'use_cache_voxels': False,

    'ref_ligand_move': [0,0,0],  # move GPCR ligand out of reference pocket
    'movie_mode': False
}
