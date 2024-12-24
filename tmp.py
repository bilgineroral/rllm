import numpy as np
import random

def mask_tokens(sequence: str, 
                vocab: list = None
                ) -> tuple[str, np.ndarray, list]:
    """
    Apply the 80/10/10 rule to mask tokens in a sequence.
    
    Returns:
        tuple: (masked_sequence, mask_indices, ground_truth_tokens)
            - masked_sequence: string with tokens masked
            - mask_indices: numpy array of masked positions
            - ground_truth_tokens: list of original tokens at masked positions
    """
    if vocab is None:
        vocab = [
            'A', 'C', 'G', 'U', 'R', 'Y', 'K', 
            'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N'
        ]
        
    seq_length = len(sequence)
    num_to_mask = random.randint(1, seq_length)
    mask_indices = np.random.choice(seq_length, num_to_mask, replace=False)
    mask_indices = np.sort(mask_indices)
    
    # Track ground truth tokens
    ground_truth_tokens = [sequence[idx] for idx in mask_indices]
    
    masked_sequence = list(sequence)
    for idx in mask_indices:
        prob = random.random()
        if prob < 0.8:  # 80%: Replace with [MASK]
            masked_sequence[idx] = "-"
        elif prob < 0.9:  # 10%: Replace with random token
            masked_sequence[idx] = random.choice(vocab)
        # else: 10% unchanged

    return "".join(masked_sequence), mask_indices, ground_truth_tokens

sequence = "AUGUCAUCUG"
a, b, c = mask_tokens(sequence)

print(sequence)
print(a)
print(b)
print(c)
