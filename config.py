import os

discrete_trans = 100.0
discrete_theta = 100.0
path = os.path.dirname(os.path.abspath(__file__)) + "/resources/gpcr"
# path = ""
config = {
    'discrete' : False,
    'K_trans' : 2,
    'K_theta' : 2,
    'normalize' : True,
    'discrete_theta' : discrete_theta,
    'discrete_trans' : discrete_trans,
    'action_space_d' : (2, 2, 2),
    'action_space_r' : (2, 2, 2, 2, 2, 2),
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
    'decay' : 0.93, # ^25 = 0.001,
    'voxel_method' : 'C',
    'debug' : True,

    'random' : None, # randomly place ligand around protein
    'many_ligands' : False, # use many ligands from the random_ligand_folder
    'random_ligand_folder': path + '/rligands',
    'random_ligand_folder_test': path + '/rligands_eval', #use train_ligands() or eval_ligands() to switch, default train_ligands called if manyligands not false
    'random_dcd' : False, # use random protein states from folder
    'protein_state_folder': '/Users/austin/gpcr/structs/',  #*.pdbs used randomly
    'load_num' : 3,  # used for speed, set number of states each env uses for training.
    'cache' : 'cache/',
    'use_cache_voxels' : False,

    'ref_ligand_move' : [0, 0, 15], #move GPCR ligand out of reference pocket
    'movie_mode' : False
}
