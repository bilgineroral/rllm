import torch

class RNATokenizer:
    def __init__(self):
        """
        Initializes the RNA tokenizer with token-to-index and index-to-token mappings.
        """
        self.vocab = {
            "<CLS>": 0,    # Start token
            "<PAD>": 1,    # Padding token
            "<EOS>": 2,    # End token
            "<UNK>": 3,    # Unknown token
            "A": 5, "a": 5,
            "C": 7, "c": 7,
            "G": 4, "g": 4,
            "U": 6, "u": 6,
            "T": 6, "t": 6
        }
        self.vocab_size = len(set(self.vocab.values()))
        self.cls_token = 0  # <CLS>
        self.pad_token = 1  # <PAD>
        self.eos_token = 2  # <EOS>
        self.unk_token = 3  # <UNK>

    def __call__(self, sequence: str):
        """
        Tokenizes a single RNA sequence into token indices.

        Args:
            sequence (str): RNA sequence (e.g., "cggccu").

        Returns:
            tokens (list): Tokenized sequence with CLS and EOS tokens added.
        """
        tokens = [self.cls_token]  # Add CLS token
        for char in sequence:
            tokens.append(self.vocab.get(char, self.unk_token))  # Map character to token or use UNK
        tokens.append(self.eos_token)  # Add EOS token
        return tokens
    
    def decode(self, tokens: list, remove_special_tokens: bool = False):
        """
        Decodes a list of token indices into a string RNA sequence.

        Args:
            tokens (list): List of token indices.

        Returns:
            sequence (str): Decoded RNA sequence.
        """
        special_tokens = {self.cls_token, self.eos_token, self.pad_token, self.unk_token}
        sequence = []
        for token in tokens:
            if remove_special_tokens and token in special_tokens:
                continue
            char = list(self.vocab.keys())[list(self.vocab.values()).index(token)]
            sequence.append(char)
        return "".join(sequence)

    def pad_sequences(self, sequences: list, pad_value: int = None):
        """
        Pads a list of tokenized sequences to the same length.

        Args:
            sequences (list of list): List of tokenized RNA sequences.
            pad_value (int): Padding value. Defaults to tokenizer's pad_token (1).

        Returns:
            padded_sequences (torch.Tensor): Padded tensor of shape [batch_size, max_seq_len].
            lengths (list): List of original sequence lengths before padding.
        """
        if pad_value is None:
            pad_value = self.pad_token

        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded_sequences = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)

        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return padded_sequences, lengths
