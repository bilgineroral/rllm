#!/bin/bash

# Check if a GitHub URL is provided as an argument
# Store the GitHub URL
> code

# Define an array of popular code file extensions
EXTENSIONS=("py" "js" "ts" "jsx" "tsx" "rs" "ex" "exs" "go" "java" "c" "cpp" "h" "hpp" \
"cs" "rb" "php" "html" "css" "kt" "swift" "scala" "sh" "pl" "r" "lua" "m" "erl" "hs")

# Build the find command arguments to search for files with the specified extensions
FIND_ARGS=()
for EXT in "${EXTENSIONS[@]}"; do
    FIND_ARGS+=( -iname "*.$EXT" -o )
done
# Remove the last '-o' (logical OR) operator
unset 'FIND_ARGS[${#FIND_ARGS[@]}-1]'

# Find all files matching the extensions and process them
find . -type f \( "${FIND_ARGS[@]}" \) | while read -r FILE; do
    # Append the filename to the 'code' file
    echo "$FILE" >> code
    # Append the file content to the 'code' file
    cat "$FILE" >> code
done
