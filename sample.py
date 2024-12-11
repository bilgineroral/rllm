import torch
import torch.nn.functional as F
from tokenizer import RNATokenizer
from model import ParallelizedCrossAttentionModel

def generate_rna_sequence(model, protein_emb, tokenizer, max_length=100, temperature=1.0, top_k=5, device="cuda"):
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
    protein_emb = protein_emb.to(device)

    # Initialize with <CLS> token
    input_tokens = [tokenizer.cls_token]
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)  # [1, 1]

    # Initial masks
    protein_mask = torch.zeros(1, protein_emb.shape[1], dtype=torch.bool).to(device)  # No protein padding
    rna_mask = None  # No RNA mask at the beginning

    # Start generating tokens
    generated_tokens = []
    print("Generated RNA Sequence: ", end="", flush=True)
    for _ in range(max_length):
        # Embed the current input tokens
        # rna_emb = tokenizer.tokens_to_embeddings(input_tensor, device)  # [1, num_layers, seq_len, d_rna]
        rna_emb = torch.randn(1, 12, len(input_tokens), 768).to(device)  # Example random RNA embedding

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
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Decode and print the latest token
        print(tokenizer.decode([next_token]), end="", flush=True)

    # Decode tokens to RNA sequence
    rna_sequence = tokenizer.decode(generated_tokens)
    print()  # Newline after the sequence is fully generated
    return rna_sequence


# Example usage
if __name__ == "__main__":
    # Load tokenizer
    tokenizer = RNATokenizer()

    # Load pretrained model
    model = ParallelizedCrossAttentionModel(
        d_rna=768, d_protein=1536, d_model=768, num_heads=8, num_layers=12, vocab_size=tokenizer.vocab_size
    )
    # model.load_state_dict(torch.load("model_checkpoint.pt"))  # Replace with your checkpoint path
    model.to("cuda")

    # Load protein embedding
    # protein_embedding = torch.load("protein_embedding.pt")  # Replace with a valid precomputed embedding file
    protein_embedding = torch.randn(100, 1536)  # Example random protein embedding
    protein_embedding = protein_embedding.unsqueeze(0)  # Add batch dimension: [1, prot_len, d_protein]

    # Generate RNA sequence
    generated_rna = generate_rna_sequence(
        model=model,
        protein_emb=protein_embedding,
        tokenizer=tokenizer,
        max_length=100,
        temperature=0.8,
        top_k=5,
        device="cuda"
    )
    print("Generated RNA Sequence:", generated_rna)