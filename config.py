import os
import math

path = os.path.dirname(os.path.abspath(__file__)) + "/resources/gpcr"
# path = ""
config = {
    'discrete' : False,
    'K_trans' : 4,
    'K_theta' : 4,
    'normalize' : False,
    'action_space_d' : (2, 2, 2),
    'action_space_r' : (2 * math.pi, 2 * math.pi, 2 * math.pi),
    'protein_wo_ligand' :  path + '/test3.pdb',
    'ligand' : path + '/gpcr_ligand.pdb',
    'oe_box' : None,
    'bp_dimension': [40, 40, 40],
    'bp_centers' : [43.31, 41.03, 77.37],
    'bp_min' : [23.31, 21.030, 57.37],
    'bp_max' : [63.31, 61.03, 97.37],
    'voxelsize' : 1.0,
    'output_size' : (40, 40, 40, 8), # (39,40,42,8),
    'max_steps' : 100,
    'decay' : 0.94, # ^25 = 0.001,
    'voxel_method' : 'C',
    'debug' : False,

    ## Reward function tuning
    'overlap_weight' : 0.01,
    'l2_decay' : 0.01,
    'dockscore_weight' : 1.0,

    ## Ligand and Protein selection features
    'random' : 0.33, # randomly place ligand around protein
    'many_ligands' : True, # use many ligands from the random_ligand_folder
    'random_ligand_folder': path + '/rligands',
    'random_ligand_folder_test': path + '/rligands_eval', #use train_ligands() or eval_ligands() to switch, default train_ligands called if manyligands not false
    'random_dcd' : False, # use random protein states from folder
    'protein_state_folder': '/Users/austin/gpcr/structs/',  #*.pdbs used randomly
    'load_num' : 3,  # used for speed, set number of states each env uses for training.
    'cache' : 'cache/',
    'use_cache_voxels' : False,

    'ref_ligand_move' : [0, 0, 10], #move GPCR ligand out of reference pocket
    'movie_mode' : False
}
