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
    'action_space_r' : (math.pi, math.pi, math.pi),
    'protein_wo_ligand' :  path + '/6dpz/6pdz_wo_ligand.pdb',
    'ligand' : path + '/6dpz/6dpz_ligand.pdb',
    'oe_box' : None,
    'bp_dimension': [36, 36, 36],
    'bp_centers' : [ 30.748,   6.935,  31.502],
    'bp_min' : [12.748, 24.935, 13.502],
    'bp_max' : [48.748, -12.935, 39.502],
    'voxelsize' : 1.0,
    'output_size' : (36, 36, 36, 8), # (39,40,42,8),
    'max_steps' : 100,
    'decay' : 0.97, # ^25 = 0.001,
    'voxel_method' : 'C',
    'debug' : False,

    ## Reward function tuning
    'overlap_weight' : 0.1,
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

    'ref_ligand_move' : [0, 0, 0], #move GPCR ligand out of reference pocket
    'movie_mode' : False
}
