import os
import math

path = os.path.dirname(os.path.abspath(__file__)) + "/resources"
# path = ""
config = {
    'discrete' : False,
    'K_trans' : 4,
    'K_theta' : 4,
    'normalize' : True,
    'action_space_d' : (2,2,2),
    'action_space_r' : (math.pi * 2 * 0.5, math.pi * 2 * 0.5, math.pi * 2 * 0.5),
    'protein_wo_ligand' :  path + '/6dpz/6pdz_wo_ligand.pdb',
    'ligand' : path + '/6dpz/6dpz_ligand.pdb',
    'oe_box' : None,
    'bp_dimension': [18, 18, 18],
    'bp_centers' : [ 30.748,   0.328,  23.421],
    'bp_min' : [21.748, -8.672, 14.421],
    'bp_max' : [39.748, 9.328, 32.421],
    'voxelsize' : 0.5,
    'output_size' : (36, 36, 36, 8), # (39,40,42,8),
    'max_steps' : 50,
    'decay' : 0.99, # ^25 = 0.001,
    'voxel_method' : 'C',
    'debug' : False,
    'reward_ramp' : 1.0,

    ## Reward function tuning
    'overlap_weight' : 0.01,
    'l2_decay' : 0.01,
    'dockscore_weight' : 1.0,

    ## Ligand and Protein selection features
    'random' : 0.25, # randomly place ligand around protein
    'many_ligands' : False, # use many ligands from the random_ligand_folder
    'random_ligand_folder': path + '/rligands',
    'random_ligand_folder_test': path + '/rligands_eval', #use train_ligands() or eval_ligands() to switch, default train_ligands called if manyligands not false
    'random_dcd' : False, # use random protein states from folder
    'protein_state_folder': '/Users/austin/gpcr/structs/',  #*.pdbs used randomly
    'load_num' : 3,  # used for speed, set number of states each env uses for training.
    'cache' : 'cache/',
    'use_cache_voxels' : False,

    'ref_ligand_move' : None, #move GPCR ligand out of reference pocket
    'movie_mode' : False
}
