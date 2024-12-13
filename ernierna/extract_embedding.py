import os
import time
import torch
import argparse
import numpy as np

from .src.ernie_rna.tasks.ernie_rna import *
from .src.ernie_rna.models.ernie_rna import *
from .src.ernie_rna.criterions.ernie_rna import *
from .src.utils import ErnieRNAOnestage, read_text_file, load_pretrained_ernierna, prepare_input_for_ernierna


def seq_to_index(sequences, add_cls=True, add_eos=True):
    '''
    Convert RNA sequences into indexed form.

    Args:
        sequences: list of strings (RNA sequences with different lengths)
        add_cls: Bool, whether to add the 'cls' token at the start
        add_eos: Bool, whether to add the 'eos' token at the end

    Returns:
        rna_index: numpy array, shape: [len(sequences), max_seq_len + add_cls + add_eos]
        rna_len_lst: list of sequence lengths
    '''
    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    seq_nums = len(rna_len_lst)

    # Initialize matrix, considering CLS and EOS tokens
    total_len = max_len + add_cls + add_eos
    rna_index = np.ones((seq_nums, total_len))

    for i, seq in enumerate(sequences):
        pos = 1 if add_cls else 0  # Starting position
        for j, char in enumerate(seq):
            if char in set("Aa"):
                rna_index[i][pos + j] = 5
            elif char in set("Cc"):
                rna_index[i][pos + j] = 7
            elif char in set("Gg"):
                rna_index[i][pos + j] = 4
            elif char in set("TUtu"):
                rna_index[i][pos + j] = 6
            else:
                rna_index[i][pos + j] = 3
        if add_eos:
            rna_index[i][pos + len(seq)] = 2  # 'eos' token
        if add_cls:
            rna_index[i][0] = 0  # 'cls' token

    return rna_index, rna_len_lst

# (my_model, sequences, if_cls=False, device='cuda', layer_idx=12
def extract_embedding_of_ernierna(my_model, sequences, if_cls=True, add_cls=True, add_eos=True, device='cuda', layer_idx=12):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_index(sequences, add_cls=add_cls, add_eos=add_eos)
    layer_num = 1 if layer_idx < 12 else layer_idx
    # extract embedding one by one
    if if_cls:
        embedding = np.zeros((len(sequences),layer_num,768))
    else:
        embedding = np.zeros((len(sequences), layer_num, max(rna_len_lst) + add_cls + add_eos, 768))

    my_model.eval()
    with torch.no_grad():
        for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
            
            one_d, two_d = prepare_input_for_ernierna(index, seq_len + add_cls + add_eos)
            one_d = one_d.to(device)
            two_d = two_d.to(device)
            
            output = my_model(one_d,two_d,layer_idx=layer_idx).cpu().detach().numpy()
            if if_cls:
                embedding[i,:,:] = output[:,0,0,:]
            else:
                embedding[i,:,:seq_len + add_cls + add_eos,:] = output[:,0,:,:]
        
    return torch.from_numpy(embedding)
