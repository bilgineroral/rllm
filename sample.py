import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
from model import RLLM
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

def sample(
        model: RLLM,
        tokenizer: AutoTokenizer,
        protein: torch.Tensor,
        idx: torch.Tensor, 
        max_new_tokens: int = 750, 
        temperature: int = 1.0, 
        device: str | torch.device = "cpu",
        top_k: int = 20,
        repetition_penalty: float = 1.0
        ):
    """
    Sample a RNA sequence from the model.
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()

    for i in range(max_new_tokens):
        logits = model(protein, idx)
        logits = logits[:, -1, :] / temperature # [B; vocab_size]

        logits, indices = torch.topk(logits, min(top_k, logits.size(-1))) # [B; top_k]
        probs = F.softmax(logits, dim=-1) # [B; top_k]

        idx_next = torch.multinomial(probs, num_samples=1)
        idx_next = torch.gather(indices, dim=-1, index=idx_next) # [B; 1]

        idx = torch.cat((idx, idx_next), dim=-1)

        if idx_next == tokenizer.eos_token_id:
            break

    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--weights", type=str, default="./checkpoint.pt")
    parser.add_argument("--decoder_weights", type=str, default="./decoder_checkpoint.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start", type=str, default="<|endoftext|>")
    parser.add_argument("--source_protein_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("Initializing model...")
    model = RLLM(gpt_checkpoint_path=args.decoder_weights)
    model.to(device)

    # print("Before: ", model.gpt.transformer.h[23].attn.c_attn.weight[:5])

    model_state_dict = torch.load(args.weights)["model_state"]
    # pytorch compiled model state dict keys have "_orig_mod." prefix so we have to remove them
    for key in list(model_state_dict.keys()):
        model_state_dict[key.replace("_orig_mod.", "")] = model_state_dict.pop(key)
    print("Loading model weights...")
    model.load_state_dict(model_state_dict)
    print("Model loaded successfully!")

    # print("After: ", model.gpt.transformer.h[23].attn.c_attn.weight[:5])

    protein = torch.load(args.source_protein_path)
    protein = protein.to(device)

    start_ids = tokenizer.encode("".join(args.start))
    tokens = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]) # add batch dim
    
    print("Sampling RNA sequence...")
    generated_rna = sample(
        model=model,
        tokenizer=tokenizer,
        protein=protein,
        idx=tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty
    )

    print(generated_rna)
