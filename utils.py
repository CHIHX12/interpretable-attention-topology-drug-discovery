import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

# Amino acid physicochemical features: (hydrophobicity, volume, charge, polarity)
AA_FEATURES = {
    "A": (1.8, 88.6, 0.0, 0.0),
    "C": (2.5, 108.5, 0.0, 0.0),
    "D": (-3.5, 111.1, -1.0, 1.0),
    "E": (-3.5, 138.4, -1.0, 1.0),
    "F": (2.8, 189.9, 0.0, 0.0),
    "G": (-0.4, 60.1, 0.0, 0.0),
    "H": (-3.2, 153.2, 1.0, 1.0),
    "I": (4.5, 166.7, 0.0, 0.0),
    "K": (-3.9, 168.6, 1.0, 1.0),
    "L": (3.8, 166.7, 0.0, 0.0),
    "M": (1.9, 162.9, 0.0, 0.0),
    "N": (-3.5, 114.1, 0.0, 1.0),
    "P": (-1.6, 112.7, 0.0, 0.0),
    "Q": (-3.5, 143.8, 0.0, 1.0),
    "R": (-4.5, 173.4, 1.0, 1.0),
    "S": (-0.8, 89.0, 0.0, 1.0),
    "T": (-0.7, 116.1, 0.0, 1.0),
    "V": (4.2, 140.0, 0.0, 0.0),
    "W": (-0.9, 227.8, 0.0, 0.0),
    "Y": (-1.3, 193.6, 0.0, 1.0),
}
AA_FEATURE_DIM = 4


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    # Unpack batch, which now contains 4 elements (d, p, y, z)
    if len(x[0]) == 4:
        d, p, y, z = zip(*x)
    else:
        # Backward compatibility: if only 3 elements, use y for z
        d, p, y = zip(*x)
        z = y

    d = dgl.batch(d)

    # Check whether p contains features (tuple format)
    if isinstance(p[0], tuple):
        # Format with features: (protein_idx, protein_feat, protein_len)
        p_idx, p_feat, p_len = zip(*p)
        return (
            d,
            (
                torch.tensor(np.array(p_idx)),      # [batch, seq_len]
                torch.tensor(np.array(p_feat)),     # [batch, seq_len, 4]
                torch.tensor(np.array(p_len))       # [batch]
            ),
            torch.tensor(y),
            torch.tensor(z)
        )
    else:
        # Original format: protein_idx only
        return d, torch.tensor(np.array(p)), torch.tensor(y), torch.tensor(z)


def graph_collate_func_with_support(batch):
    """
    Collate function for batches with support sets (BatLiNet-style).

    Input: list of dicts with 'query' and 'support' keys
    Output: batched query and support data
    """
    # Separate query and support
    queries = [item['query'] for item in batch]
    supports = [item['support'] for item in batch]

    # Collate queries (standard way)
    # Unpack batch, which now contains 4 elements (d, p, y, z)
    if len(queries[0]) == 4:
        d_query, p_query, y_query, z_query = zip(*queries)
    else:
        # Backward compatibility: if only 3 elements, use y for z
        d_query, p_query, y_query = zip(*queries)
        z_query = y_query

    d_query = dgl.batch(d_query)

    # Handle protein data
    if isinstance(p_query[0], tuple):
        p_idx, p_feat, p_len = zip(*p_query)
        p_query = (
            torch.tensor(np.array(p_idx)),
            torch.tensor(np.array(p_feat)),
            torch.tensor(np.array(p_len))
        )
    else:
        p_query = torch.tensor(np.array(p_query))

    y_query = torch.tensor(y_query)
    z_query = torch.tensor(z_query)

    # Collate supports
    # supports is a list of B items, each containing (list_of_K_drugs, list_of_K_proteins, list_of_K_labels, list_of_K_z)
    B = len(supports)
    K = len(supports[0][0])  # Number of support samples

    # Flatten all support drugs: B*K graphs
    support_drugs_flat = []
    for b in range(B):
        for k in range(K):
            support_drugs_flat.append(supports[b][0][k])
    d_support = dgl.batch(support_drugs_flat)  # Batch all B*K graphs

    # Flatten and batch support proteins
    if isinstance(supports[0][1][0], tuple):
        # BiLSTM mode with features
        support_proteins_idx = []
        support_proteins_feat = []
        support_proteins_len = []
        for b in range(B):
            for k in range(K):
                support_proteins_idx.append(supports[b][1][k][0])
                support_proteins_feat.append(supports[b][1][k][1])
                support_proteins_len.append(supports[b][1][k][2])
        p_support = (
            torch.tensor(np.array(support_proteins_idx)),  # [B*K, seq_len]
            torch.tensor(np.array(support_proteins_feat)),  # [B*K, seq_len, 4]
            torch.tensor(np.array(support_proteins_len))  # [B*K]
        )
    else:
        # CNN mode
        support_proteins_flat = []
        for b in range(B):
            for k in range(K):
                support_proteins_flat.append(supports[b][1][k])
        p_support = torch.tensor(np.array(support_proteins_flat))  # [B*K, seq_len]

    # Support labels: [B, K]
    y_support = []
    z_support = []
    for b in range(B):
        y_support.append(supports[b][2])
        # Handle z values (if support set has 4 elements)
        if len(supports[b]) == 4:
            z_support.append(supports[b][3])
        else:
            # Backward compatibility: use y as z
            z_support.append(supports[b][2])
    y_support = torch.tensor(y_support)  # [B, K]
    z_support = torch.tensor(z_support)  # [B, K]

    return {
        'query': (d_query, p_query, y_query, z_query),
        'support': (d_support, p_support, y_support, z_support)
    }


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


def encode_sequence_with_features(sequence, max_length=1200):
    """
    Encode a protein sequence and extract physicochemical features.
    Args:
        sequence (str): Protein sequence string
        max_length (int): Maximum sequence length
    Returns:
        encoding: Amino acid ID encoding [max_length]
        feats: Physicochemical features [max_length, 4]
        length: Actual sequence length
    """
    encoding = np.zeros(max_length)
    feats = np.zeros((max_length, AA_FEATURE_DIM))
    length = min(len(sequence or ""), max_length)

    for idx, letter in enumerate((sequence or "")[:max_length]):
        letter = letter.upper()
        encoding[idx] = CHARPROTSET.get(letter, 0)
        feats[idx] = AA_FEATURES.get(letter, (0.0, 0.0, 0.0, 0.0))

    if length <= 0:
        length = 1

    return encoding, feats, length


# ===== SELFIES-related utilities =====

def smiles_to_selfies(smiles):
    """
    Convert a SMILES string to SELFIES.

    Args:
        smiles: SMILES string

    Returns:
        selfies: SELFIES string, or None if conversion fails
    """
    try:
        import selfies as sf
        return sf.encoder(smiles)
    except Exception as e:
        logging.warning(f"Failed to convert SMILES to SELFIES: {smiles}, error: {e}")
        return None


def build_selfies_vocab(smiles_list, max_vocab_size=None):
    """
    Build a SELFIES vocabulary from a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        max_vocab_size: Maximum vocabulary size (optional)

    Returns:
        vocab: Dictionary mapping {selfies_token: idx}, index 0 reserved for padding
    """
    import selfies as sf
    from collections import Counter

    all_tokens = Counter()

    for smiles in smiles_list:
        try:
            selfies_str = sf.encoder(smiles)
            tokens = list(sf.split_selfies(selfies_str))
            all_tokens.update(tokens)
        except Exception as e:
            logging.warning(f"Failed to process SMILES: {smiles}, error: {e}")
            continue

    # Sort by frequency
    sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)

    if max_vocab_size:
        sorted_tokens = sorted_tokens[:max_vocab_size-1]  # Reserve 0 for padding

    # Build vocab (0 reserved for padding)
    vocab = {token: idx+1 for idx, (token, _) in enumerate(sorted_tokens)}

    return vocab


def encode_selfies(smiles, vocab, max_length=200):
    """
    Convert a SMILES string to SELFIES and encode as an integer sequence.

    Args:
        smiles: SMILES string
        vocab: SELFIES vocabulary ({token: idx})
        max_length: Maximum sequence length

    Returns:
        encoding: Encoded integer array [max_length]
        length: Actual sequence length
    """
    import selfies as sf

    encoding = np.zeros(max_length, dtype=np.int64)

    try:
        selfies_str = sf.encoder(smiles)
        tokens = list(sf.split_selfies(selfies_str))
        length = min(len(tokens), max_length)

        for idx, token in enumerate(tokens[:max_length]):
            encoding[idx] = vocab.get(token, 0)  # Unknown tokens map to 0 (padding)

    except Exception as e:
        logging.warning(f"Failed to encode SMILES: {smiles}, error: {e}")
        length = 0

    if length <= 0:
        length = 1  # Avoid pack_padded_sequence error

    return encoding, length


def extract_drug_features(smiles):
    """
    Extract drug physicochemical features from a SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        features: numpy array [8] containing:
            - MW (Molecular Weight)
            - logP (Lipophilicity)
            - TPSA (Topological Polar Surface Area)
            - HBD (Hydrogen Bond Donors)
            - HBA (Hydrogen Bond Acceptors)
            - Ring Count
            - Rotatable Bonds
            - Formal Charge
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Failed to parse SMILES: {smiles}")
            return np.zeros(8, dtype=np.float32)

        features = np.array([
            Descriptors.MolWt(mol),              # MW
            Descriptors.MolLogP(mol),            # logP
            Descriptors.TPSA(mol),               # TPSA
            Lipinski.NumHDonors(mol),            # HBD
            Lipinski.NumHAcceptors(mol),         # HBA
            Lipinski.RingCount(mol),             # Ring count
            Lipinski.NumRotatableBonds(mol),     # Rotatable bonds
            Chem.GetFormalCharge(mol)            # Formal charge
        ], dtype=np.float32)

        return features

    except Exception as e:
        logging.warning(f"Failed to extract features from SMILES: {smiles}, error: {e}")
        return np.zeros(8, dtype=np.float32)


DRUG_FEATURE_DIM = 8  # Drug physicochemical feature dimension


def encode_selfies_with_features(smiles, vocab, max_length=200):
    """
    Convert a SMILES string to a SELFIES encoding and extract drug physicochemical features.

    Args:
        smiles: SMILES string
        vocab: SELFIES vocabulary ({token: idx})
        max_length: Maximum sequence length

    Returns:
        encoding: SELFIES encoded integer array [max_length]
        features: Drug physicochemical features [8]
        length: Actual sequence length
    """
    # Encode SELFIES
    encoding, length = encode_selfies(smiles, vocab, max_length)

    # Extract physicochemical features
    features = extract_drug_features(smiles)

    return encoding, features, length
