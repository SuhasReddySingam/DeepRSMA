from data.m import MoleculeEmbedder
from data.process_data_rna import RNA_dataset
from main_cv import DeepRSMA
import torch
device="mps"
model=DeepRSMA()
model.load_state_dict(torch.load("/Users/suhasreddy/Downloads/model_All_sf2_5_1.pth",map_location=torch.device(device)))
molecule="CCO"
rna="AUGCUAG"
embedder=MoleculeEmbedder(vocab_path='data/smiles_vocab.pkl')
mol_emb=embedder.embed_molecule(molecule)
print(mol_emb)
