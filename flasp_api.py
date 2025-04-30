from flask import Flask, request, jsonify
import torch
from data.m import MoleculeEmbedder
from data.r import RNA_Processor
from data.vocab import WordVocab  # Added to fix pickle loading
from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch.nn as nn
import os
hidden_dim = 128
device="cuda"
# Define DeepRSMA model (unchanged)
class DeepRSMA(nn.Module):
    def __init__(self):
        super(DeepRSMA, self).__init__()
        hidden_dim = 128
        self.rna_graph_model = RNA_feature_extraction(hidden_dim)
        self.mole_graph_model = GNN_molecule(hidden_dim)
        self.mole_seq_model = mole_seq_model(hidden_dim)
        self.cross_attention = cross_attention(hidden_dim)
        self.line1 = nn.Linear(hidden_dim*2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        self.rna1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.rna2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, rna_batch, mole_batch):
        rna_out_seq, rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, device)
        mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)
        mole_seq_final = (mole_seq_emb[-1]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)

        flag = 0
        mole_out_graph = []
        mask = []
        if isinstance(mole_batch.graph_len, int):
            mole_batch.graph_len = [mole_batch.graph_len]
        elif isinstance(mole_batch.graph_len, torch.Tensor):
            mole_batch.graph_len = mole_batch.graph_len.tolist()
        for i in mole_batch.graph_len:
            count_i = i
            x = mole_graph_emb[flag:flag+count_i]
            temp = torch.zeros((128-x.size()[0]), hidden_dim).to(device)
            x = torch.cat((x, temp),0)
            mole_out_graph.append(x)
            mask.append([] + count_i * [1] + (128 - count_i) * [0])
            flag += count_i
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)
        context_layer, attention_score = self.cross_attention([rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph], [rna_mask_seq.to(device), rna_mask_graph.to(device), mole_mask_seq.to(device), mole_mask_graph.to(device)], device)
        
        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]
        
        rna_cross_seq = ((out_rna[:, 0:512]*(rna_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_seq_final ) / 2
        rna_cross_stru = ((out_rna[:, 512:]*(rna_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_graph_final) / 2        
        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout((self.relu(self.rna1(rna_cross)))))
        
        mole_cross_seq = ((out_mole[:,0:128]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_seq_final) / 2
        mole_cross_stru = ((out_mole[:,128:]*(mole_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_graph_final) / 2
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout((self.relu(self.mole1(mole_cross)))))   
        
        out = torch.cat((rna_cross, mole_cross),1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        return out

# Initialize Flask app
app = Flask(__name__)

# Set device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processors
try:
    model = DeepRSMA().to(device)
    model.load_state_dict(torch.load("save/model_independent_1.pth", map_location=device))
    model.eval()
    embedder = MoleculeEmbedder(vocab_path='data/smiles_vocab.pkl')
    rna_processor = RNA_Processor()
except Exception as e:
    print(f"Error loading model or processors: {e}")
    raise

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate JSON input
        if not request.json or 'rna' not in request.json or 'molecule' not in request.json:
            return jsonify({'error': 'Missing "rna" or "molecule" in JSON payload'}), 400
        
        rna = request.json['rna']
        molecule = request.json['molecule']
        
        # Validate inputs
        if not isinstance(rna, str) or not rna:
            return jsonify({'error': 'RNA must be a non-empty string'}), 400
        if not isinstance(molecule, str) or not molecule:
            return jsonify({'error': 'Molecule must be a non-empty SMILES string'}), 400
        
        # Process inputs
        try:
            mol_emb = embedder.embed_molecule(molecule)
            # rna_processor.rna_sequence = rna
            rna_emb = rna_processor.process(rna)
        except Exception as e:
            return jsonify({'error': f'Error processing inputs: {str(e)}'}), 400
        
        # Run inference
        with torch.no_grad():
            prediction = model(rna_emb.to(device), mol_emb.to(device)).item()
        
        return jsonify({'prediction': prediction}), 200
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Error handling for invalid routes
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)