#!/bin/bash

# List of IDs
ids=("1" "2" "3" "4")  # Modify this list as needed

# Max parallel processes
num_processes=4

# Run watermark_generator.py in parallel
parallel -j $num_processes python watermarked_text_generator.py ::: "${ids[@]}"

echo "All watermarking jobs completed."