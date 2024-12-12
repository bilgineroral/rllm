import random
from collections import defaultdict

# Input and output file paths
INPUT_FILE = "dataset/pairs.txt"
TRAIN_FILE = "dataset/train.txt"
VAL_FILE = "dataset/val.txt"
TEST_FILE = "dataset/test.txt"

# Proportions for the splits
TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
TEST_RATIO = 0.10

# Step 1: Read and group data by protein chain
def read_and_group_data(input_file):
    groups = defaultdict(list)
    with open(input_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                header = lines[i].strip()
                sequence = lines[i + 1].strip()
                # Extract PDB_ID and protein chain (first two fields)
                group_key = header.split('_')[0] + '_' + header.split('_')[1]
                groups[group_key].append((header, sequence))
                i += 2
            else:
                i += 1
    return groups

# Step 2: Split data into train, val, and test
def split_data(groups):
    all_keys = list(groups.keys())
    random.shuffle(all_keys)  # Shuffle keys to randomize the data

    # Calculate sizes for each split
    total = len(all_keys)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    # Create splits
    train_keys = all_keys[:train_size]
    val_keys = all_keys[train_size:train_size + val_size]
    test_keys = all_keys[train_size + val_size:]

    return train_keys, val_keys, test_keys

# Step 3: Write the data to files
def write_splits(groups, train_keys, val_keys, test_keys):
    def write_to_file(file, keys):
        with open(file, 'w') as f:
            for key in keys:
                for header, sequence in groups[key]:
                    f.write(f"{header}\n{sequence}\n")

    write_to_file(TRAIN_FILE, train_keys)
    write_to_file(VAL_FILE, val_keys)
    write_to_file(TEST_FILE, test_keys)

# Main function to orchestrate the process
def main():
    print("Reading and grouping data...")
    groups = read_and_group_data(INPUT_FILE)
    print(f"Total unique protein chains: {len(groups)}")

    print("Splitting data into train, val, and test...")
    train_keys, val_keys, test_keys = split_data(groups)

    print(f"Train size: {len(train_keys)} chains")
    print(f"Validation size: {len(val_keys)} chains")
    print(f"Test size: {len(test_keys)} chains")

    print("Writing splits to files...")
    write_splits(groups, train_keys, val_keys, test_keys)
    print("Done! Files created:")
    print(f"- {TRAIN_FILE}\n- {VAL_FILE}\n- {TEST_FILE}")

if __name__ == "__main__":
    main()

