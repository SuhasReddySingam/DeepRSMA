# Install dependencies in Colab
!apt-get update
!apt-get install -y g++ make wget unzip
!pip install nglview biopython rdkit torch torch-geometric matplotlib seaborn numpy

# Install RNAstructure
!wget http://rna.urmc.rochester.edu/RNAstructureSource/RNAstructureSource-6.4.tar.gz
!tar -xzf RNAstructureSource-6.4.tar.gz
%cd RNAstructure
!make
!make install
%cd ..

# Set RNAstructure environment
import os
os.environ['DATAPATH'] = '/content/RNAstructure/data_tables'

# Import libraries
import torch
from torch.utils.data import DataLoader
from data import RNA_dataset
from data import Molecule_dataset
from main_cv import DeepRSMA
from main_cv import CustomDualDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
import subprocess
from Bio.PDB import PDBParser
import nglview as nv

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load datasets
rna_dataset = RNA_dataset(RNA_type='All_sf')
molecule_dataset = Molecule_dataset(RNA_type='All_sf')

# Example RNA and molecule
rna_sequence = "AUGCCUAGU"
rna_data = rna_dataset.process_user_rna(rna_sequence, t_id="custom_rna_1")
mol_data = molecule_dataset[0]

# Create dataset and loader
dataset = CustomDualDataset([rna_data], [mol_data])
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
model = DeepRSMA().to(device)
model.load_state_dict(torch.load('model_All_sf2_1_1.pth', map_location=device))
model.eval()

# Extract attention weights
def get_attention_weights(model, rna_batch, mol_batch):
    with torch.no_grad():
        rna_out_seq, rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = model.rna_graph_model(rna_batch, device)
        mole_graph_emb, mole_graph_final = model.mole_graph_model(mol_batch)
        mole_seq_emb, _, mole_mask_seq = model.mole_seq_model(mol_batch, device)
        mole_seq_final = (mole_seq_emb[-1] * (mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)
        flag = 0
        mole_out_graph = []
        mask = []
        for i in mol_batch.graph_len:
            count_i = i
            x = mole_graph_emb[flag:flag + count_i]
            temp = torch.zeros((128 - x.size()[0]), 128).to(device)
            x = torch.cat((x, temp), 0)
            mole_out_graph.append(x)
            mask.append([] + count_i * [1] + (128 - count_i) * [0])
            flag += count_i
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)
        _, attention_score = model.cross_attention(
            [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
            [rna_mask_seq.to(device), rna_mask_graph.to(device), mole_mask_seq.to(device), mole_mask_graph.to(device)],
            device
        )
        return attention_score, rna_batch, mol_batch

# Compute nucleotide importance
def compute_nucleotide_importance(attention_score, rna_sequence):
    attention_score = attention_score.mean(dim=1).squeeze(0)
    nucleotide_scores = attention_score.sum(dim=1).cpu().numpy()
    nucleotide_scores = nucleotide_scores / nucleotide_scores.max()
    rna_labels = list(rna_sequence)
    return nucleotide_scores, rna_labels

# Predict RNA structure with RNAstructure
def predict_rna_3d_structure(rna_sequence, output_dir="rna_structure"):
    os.makedirs(output_dir, exist_ok=True)
    seq_file = os.path.join(output_dir, "rna_sequence.seq")
    ct_file = os.path.join(output_dir, "rna_structure.ct")
    pdb_file = os.path.join(output_dir, "rna_structure.pdb")
    
    # Write sequence file
    with open(seq_file, 'w') as f:
        f.write(f">{rna_sequence}\n{rna_sequence}\n")
    
    # Run Fold for secondary structure
    subprocess.run(["/content/RNAstructure/exe/Fold", seq_file, ct_file, "-mfe"], check=True)
    
    # Run Partition and MaxExpect for better secondary structure
    pfs_file = os.path.join(output_dir, "rna_structure.pfs")
    subprocess.run(["/content/RNAstructure/exe/Partition", seq_file, pfs_file], check=True)
    subprocess.run(["/content/RNAstructure/exe/MaxExpect", pfs_file, ct_file], check=True)
    
    # Note: 3dRNApredict is not standard in RNAstructure; use a placeholder (e.g., ct2dot + external tool)
    # For simplicity, use ct2dot to convert CT to dot-bracket and simulate 3D prediction
    dot_file = os.path.join(output_dir, "rna_structure.dot")
    subprocess.run(["/content/RNAstructure/exe/ct2dot", ct_file, "1", dot_file], check=True)
    
    # Simulate 3D prediction (replace with actual 3dRNApredict if available)
    # For demo, use a precomputed PDB or external tool like RNAComposer
    # Here, we'll use RNAComposer via web API (requires internet)
    with open(dot_file) as f:
        dot_bracket = f.read().split("\n")[2].strip()
    from urllib.request import urlopen
    import urllib.parse
    rna_composer_url = "http://rnacomposer.cs.put.poznan.pl/api/predict"
    data = urllib.parse.urlencode({"sequence": rna_sequence, "structure": dot_bracket}).encode()
    with urlopen(rna_composer_url, data) as response:
        with open(pdb_file, 'wb') as f:
            f.write(response.read())
    
    return pdb_file

# Visualize 3D structure with NGL Viewer
def visualize_3d_binding_sites(pdb_file, nucleotide_scores, rna_labels):
    # Load PDB
    view = nv.show_file(pdb_file, default_representation=False)
    
    # Add cartoon representation
    view.add_representation("cartoon", color="grey")
    
    # Color nucleotides by importance
    for i, (nuc, score) in enumerate(zip(rna_labels, nucleotide_scores)):
        r = int(score * 255)  # Red intensity
        b = int((1 - score) * 255)  # Blue intensity
        color = f"#{r:02x}00{b:02x}"
        view.add_representation("cartoon", selection=f":{i+1}", color=color)
        if score > 0.5:
            view.add_representation("ball+stick", selection=f":{i+1}", color=color)
    
    # Display
    view.center()
    view.camera = "orthographic"
    view.download_image("binding_sites_3d.png", factor=4)
    return view

# Run
for batch in loader:
    rna_batch, mol_batch = batch
    attention_score, rna_batch, mol_batch = get_attention_weights(model, rna_batch.to(device), mol_batch.to(device))
    
    # Get SMILES (placeholder)
    mol_smiles = "CC(=O)NC1=CC=CC=C1"
    
    # Compute importance
    nucleotide_scores, rna_labels = compute_nucleotide_importance(attention_score, rna_sequence)
    
    # Predict pKd
    score = model(rna_batch.to(device), mol_batch.to(device))
    pKd = score.item()
    
    # Predict 3D structure
    pdb_file = predict_rna_3d_structure(rna_sequence)
    
    # Visualize 3D structure
    view = visualize_3d_binding_sites(pdb_file, nucleotide_scores, rna_labels)
    display(view)
    
    # Print results
    print("RNA Sequence:", rna_sequence)
    print("Nucleotide Importance Scores:", [f"{nuc}: {score:.3f}" for nuc, score in zip(rna_labels, nucleotide_scores)])
    print("Highlighted Binding Sites (score > 0.5):", [nuc for nuc, score in zip(rna_labels, nucleotide_scores) if score > 0.5])
    print("Predicted pKd:", pKd)
    print("3D Structure Saved:", "binding_sites_3d.png")
    
    # Validate with contact map
    print("Contact map edges:", rna_batch.edge_index.t().cpu().numpy())