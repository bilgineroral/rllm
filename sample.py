import torch
import torch.nn.functional as F
import argparse

from tokenizer import RNATokenizer
from model import ParallelizedCrossAttentionModel

from ernierna.src.utils import ErnieRNAOnestage, load_pretrained_ernierna
from ernierna.extract_embedding import extract_embedding_of_ernierna

ERNIERNA_WEIGHTS_PATH = "ernierna/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt"


def generate_rna_sequence(model, decoder, protein_emb, tokenizer, max_length=100, temperature=1.5, top_k=5, device="cuda"):
    """
    Generates an RNA sequence given the protein embeddings.

    Args:
        model: Trained ParallelizedCrossAttentionModel.
        protein_emb: Precomputed protein embeddings [1, prot_len, d_protein].
        tokenizer: RNATokenizer instance.
        max_length: Maximum length of generated RNA sequence.
        temperature: Sampling temperature for scaling logits.
        top_k: Limits sampling to top-k logits.
        device: Device to perform computations.

    Returns:
        Generated RNA sequence (string).
    """
    model.eval()
    decoder.eval()
    protein_emb = protein_emb.to(device).float()

    # Initialize with <CLS> token
    input_tokens = [tokenizer.cls_token]

    # Initial masks
    protein_mask = torch.zeros(1, protein_emb.shape[1], dtype=torch.bool).to(device)  # No protein padding
    rna_mask = None  # No RNA mask at the beginning

    # Start generating tokens
    generated_tokens = []
    print("Generated RNA Sequence: ", end="", flush=True)
    for _ in range(max_length):
        rna_emb = extract_embedding_of_ernierna(
            decoder,
            [tokenizer.decode(input_tokens, remove_special_tokens=True)],
            if_cls=False,
            add_cls=True,   # Control CLS addition here
            add_eos=False,  # No need to add EOS token during generation
            device=device,
            layer_idx=12
        )
        # rna_emb = rna_emb.squeeze(0)  # Remove batch dimension: [12, current_length+2, 768]
        # rna_emb = rna_emb[-1, 1:-1, :]  # Take the last layer's embeddings and remove CLS and EOS tokens: [current_length, 768]
        rna_emb = rna_emb.to(device).float()
        # Forward pass
        logits = model(rna_emb, protein_emb, rna_mask, protein_mask)  # [1, seq_len, vocab_size]
        logits = logits[:, -1, :]  # Take the last token's logits: [1, vocab_size]

        # Scale logits using temperature
        logits = logits / temperature

        # Apply Top-K sampling
        if top_k > 0:
            values, indices = torch.topk(logits, k=top_k)
            logits_topk = torch.zeros_like(logits).scatter_(1, indices, values)
            logits = logits_topk

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample a token
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Stop generation if EOS token is predicted
        if next_token == tokenizer.eos_token:
            break

        # Append generated token and update input tensor
        generated_tokens.append(next_token)
        input_tokens.append(next_token)

        # Decode and print the latest token
        print(tokenizer.decode([next_token], remove_special_tokens=True), end="", flush=True)

    # Decode tokens to RNA sequence
    rna_sequence = tokenizer.decode(generated_tokens)
    print()  # Newline after the sequence is fully generated
    return rna_sequence


if __name__ == "__main__":
    # cli args
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True, type=str, help="Path to model weights")
    parser.add_argument("--protein", required=True, type=str, help="Path to protein embedding")
    parser.add_argument("--device", default="cuda", type=str, help="Device to perform computations")
    parser.add_argument("--temperature", default=1, type=float, help="Sampling temperature")
    parser.add_argument("--top_k", default=3, type=int, help="Top-K sampling")
    parser.add_argument("--max_length", default=100, type=int, help="Maximum length of generated RNA sequence")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = RNATokenizer()

    # Load pretrained model
    model = ParallelizedCrossAttentionModel(
        d_rna=768, d_protein=1536, d_model=768, num_heads=8, num_layers=12, vocab_size=tokenizer.vocab_size
    )
    model.load_state_dict(torch.load(args.pretrained)["model_state"]) 
    model.to(args.device)

    model_pretrained = load_pretrained_ernierna(ERNIERNA_WEIGHTS_PATH, arg_overrides={"data": 'ernierna/src/dict/'})
    rna_decoder = ErnieRNAOnestage(model_pretrained.encoder).to(args.device)

    # Load protein embedding
    protein_embedding = torch.load(args.protein)  # Shape: [prot_len + 2, d_protein]

    # Generate RNA sequence
    generated_rna = generate_rna_sequence(
        model=model,
        decoder=rna_decoder,
        protein_emb=protein_embedding,
        tokenizer=tokenizer,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
