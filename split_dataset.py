import os
import random

def read_entries(file_path):
    """
    Reads the input file and returns a list of entries.
    Each entry is a tuple: (header, sequence)
    """
    entries = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    if len(lines) % 2 != 0:
        raise ValueError("The input file has an odd number of lines. Each entry should consist of a header and a sequence.")
    
    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        sequence = lines[i+1].strip()
        entries.append((header, sequence))
    
    return entries

def split_data(entries, train_ratio=0.85, val_ratio=0.05, test_ratio=0.1, seed=42):
    """
    Splits the entries into training, validation, and test sets.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    
    random.seed(seed)
    random.shuffle(entries)
    
    total = len(entries)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = entries[:train_end]
    val_set = entries[train_end:val_end]
    test_set = entries[val_end:]
    
    return train_set, val_set, test_set

def write_entries(entries, file_path):
    """
    Writes the list of entries to the specified file.
    Each entry is written as two lines: header and sequence.
    """
    with open(file_path, 'w') as file:
        for header, sequence in entries:
            file.write(f"{header}\n{sequence}\n")

def main():
    input_file = './dataset/pairs.txt'
    train_file = './dataset/train.txt'
    val_file = './dataset/val.txt'
    test_file = './dataset/test.txt'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return
    
    print("Reading entries from the input file...")
    entries = read_entries(input_file)
    print(f"Total entries: {len(entries)}")
    
    print("Splitting data into train, validation, and test sets...")
    train_set, val_set, test_set = split_data(entries, seed=42)
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    print("Writing training set to train.txt...")
    write_entries(train_set, train_file)
    
    print("Writing validation set to val.txt...")
    write_entries(val_set, val_file)
    
    print("Writing test set to test.txt...")
    write_entries(test_set, test_file)
    
    print("Data splitting completed successfully.")

if __name__ == "__main__":
    main()
