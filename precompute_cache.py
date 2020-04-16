from config import config as conf
import glob as glob
from tqdm import tqdm
from rldock.environments.utils import Voxelizer, MultiScorerFromReceptor
from multiprocessing import Pool

def make_receptor( pdb):
    from openeye import oedocking, oechem
    import os.path

    file_name = str(os.path.basename(pdb))
    check_oeb = conf['cache'] + file_name.split(".")[0] + ".oeb"
    if os.path.isfile(check_oeb):
        g = oechem.OEGraphMol()
        oedocking.OEReadReceptorFile(g, check_oeb)
        return g
    else:
        proteinStructure = oechem.OEGraphMol()
        ifs = oechem.oemolistream(pdb)
        ifs.SetFormat(oechem.OEFormat_PDB)
        oechem.OEReadMolecule(ifs, proteinStructure)

        box = oedocking.OEBox(*conf['bp_max'], *conf['bp_min'])

        receptor = oechem.OEGraphMol()
        s = oedocking.OEMakeReceptor(receptor, proteinStructure, box)
        oedocking.OEWriteReceptorFile(receptor, check_oeb)
        assert (s != False)
        return receptor

def putincache(pdb_file_name):
    voxelizer = Voxelizer(pdb_file_name, conf, write_cache=True)
    recept = make_receptor(pdb_file_name)
    oe_scorer = MultiScorerFromReceptor(recept)
    assert(oe_scorer.scorers[0].IsInitialized())
    return 1

import numpy as np
import os.path
listings = glob.glob(conf['protein_state_folder'] + "*.pdb")
print("listing len", len(listings))
ordering = list(map(lambda x : int(str(os.path.basename(x)).split('.')[0].split("_")[-1]), listings))
ordering = np.argsort(ordering)[:700]
print("Making ordering....")
print(listings[0], len(listings))
listings = [listings[i] for i in ordering][:200]

res = []
with Pool(5) as p:
    its = p.imap_unordered(putincache, listings)
    for rest in tqdm(its):
        res.append(rest)