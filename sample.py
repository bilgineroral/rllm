import torch
import argparse
from model import LengthPredictionHead, RLLM
import warnings
from constants import RNA_LENGTH_CLUSTERS
import random

from fm.data import BatchConverter, Alphabet

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

def decode(sequence: torch.Tensor, alphabet: Alphabet):
    """
    Decode a sequence of indices into a string.
    """
    return "".join(['-' if alphabet.get_tok(idx) == '<mask>' 
                   else alphabet.get_tok(idx) 
                   for idx in sequence])


import torch.nn.functional as F

def mask_predict(
    rllm: RLLM,
    tokenizer: BatchConverter,
    tgt_len: int,
    protein: torch.Tensor,
    max_iters: int = None,
    device: str = "cuda"
):
    if max_iters is None:
        max_iters = tgt_len
        
    # 1) Initialize target as all masked.
    tokens = ["-"] * tgt_len
    probabilities = [0.0] * tgt_len  # track confidence per token

    # Keep a record of the sequence after each iteration
    all_predictions = []

    for t in range(max_iters):
        # 2) Convert current tokens into tensor form (tokenizer adds BOS/EOS).
        seq_str = "".join(tokens)
        data = [("tgt", seq_str)]
        _, _, tgt_idx = tokenizer(data)
        tgt_idx = tgt_idx.to(device)

        # 3) Forward pass => [1, tgt_len, vocab_size].
        logits = rllm(protein, tgt_idx)

        # 4) Update only masked positions with the top predicted token.
        for i in range(tgt_len):
            if tokens[i] == "-":
                logit_i = logits[0, i, :]
                prob_i = F.softmax(logit_i, dim=-1)
                top_id = torch.argmax(prob_i).item()
                tokens[i] = rllm.alphabet.get_tok(top_id)
                probabilities[i] = prob_i[top_id].item()

        # Record the sequence after this iteration
        current_prediction = "".join(tokens)
        all_predictions.append(current_prediction)

        # 5) Decide how many tokens to mask again for the next iteration.
        #    Example: linearly decay the masking ratio.
        n = int(tgt_len * (max_iters - (t + 1)) / max_iters)
        if n <= 0:
            break

        # 6) Identify the n positions with the lowest probability and mask them again.
        idxs_sorted = sorted(range(tgt_len), key=lambda i: probabilities[i])
        lowest = idxs_sorted[:n]

        # If desired, skip positions that exceed a certain threshold.
        # For now, we re‑mask all in 'lowest':
        for idx in lowest:
            tokens[idx] = "-"

    # Return final sequence + intermediate predictions
    final_seq = "".join(tokens)
    return final_seq, all_predictions



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length_prediction_head", type=str, default="./fm/weights/length_prediction_head.pt", help="Path to LengthPredictionHead model")
    parser.add_argument("--rllm", type=str, default="./fm/weights/rllm.pt", help="Path to RLLM model")
    parser.add_argument("--rna_fm", type=str, default="./fm/weights/rnafm.pt", help="Path to RNA-FM model")
    parser.add_argument("--max_iters", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--protein", type=str, default="protein.pt", help="Path to source protein embedding")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--trials", type=int, default=4, help="Number of trials to run (different l values to pick from)")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    length_predictor = LengthPredictionHead()
    checkpoint = torch.load(args.length_prediction_head)
    model_state_dict = checkpoint["model_state"]
    for key in list(model_state_dict.keys()):
        model_state_dict[key.replace("_orig_mod.", "")] = model_state_dict.pop(key)
    length_predictor.load_state_dict(checkpoint["model_state"])
    length_predictor.eval()
    length_predictor.to(device)

    rllm = RLLM()
    checkpoint = torch.load(args.rllm)
    model_state_dict = checkpoint["model_state"]
    for key in list(model_state_dict.keys()):
        model_state_dict[key.replace("_orig_mod.", "")] = model_state_dict.pop(key)
    rllm.load_state_dict(model_state_dict)
    rllm.eval()
    rllm.to(device)

    tokenizer = rllm.alphabet.get_batch_converter()

    protein = torch.load(args.protein).to(device) # [1, prot_len, d_protein]
    logits = length_predictor(protein) # [B, 50]

    # select top "L" RNA length clusters
    top_clusters = torch.topk(logits, args.trials, dim=1).indices[0]
    ranges = [RNA_LENGTH_CLUSTERS[idx.item()][1] for idx in top_clusters]
    lengths = [random.randint(min_len, max_len) for min_len, max_len in ranges]

    pred_targets, all_iters = [], []
    for tgt_len in lengths:
        tgt, iters = mask_predict(rllm, tokenizer, tgt_len, protein, args.max_iters, args.device)
        pred_targets.append(tgt)
        all_iters.append(iters)

best_target = None
best_score = float('-inf')

# Evaluate each final predicted sequence
for target_seq in pred_targets:
    # 1) Convert string -> tokens via tokenizer
    data = [("final_pred", target_seq)]
    _, _, tgt_idx = tokenizer(data)
    tgt_idx = tgt_idx.to(device)

    # 2) Run the model to get logits => [1, seq_len, vocab_size]
    logits = rllm(protein, tgt_idx)

    # 3) Compute average log p(token) only for the “real” tokens:
    #    Typically you skip the first (BOS) and last (EOS) if your tokenizer adds them.
    #    If tgt_idx.shape[1] = length + 2, then the real tokens are i in [1..(length)], ignoring 0, and ignoring the last index.
    sum_logprob = 0.0
    count = 0
    # Example: skip i=0 and i=(seq_len-1)
    for i in range(1, tgt_idx.shape[1] - 1):
        token_id = tgt_idx[0, i].item()
        # log_softmax over the vocab dimension
        log_probs_i = F.log_softmax(logits[0, i, :], dim=-1)
        sum_logprob += log_probs_i[token_id].item()
        count += 1

    # 4) Average log‑probability = sum_logprob / count
    avg_logprob = sum_logprob / max(count, 1)

    # 5) Keep track of which is best
    if avg_logprob > best_score:
        best_score = avg_logprob
        best_target = target_seq

print("Best target sequence:", best_target)
print("Average log‑probability:", best_score)