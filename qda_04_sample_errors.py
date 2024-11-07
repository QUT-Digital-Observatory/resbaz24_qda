import json
import os
import sys
import pandas as pd
import random
import argparse

def strip_whitespace(text):
    return ' '.join(text.split())

def process_files(directory_path, parquet_path, sample_size, output_file):
    """
    Process JSON files in the directory and match with parquet data.
    
    Args:
        directory_path (str): Path to directory containing response_*.json files
        parquet_path (str): Path to parquet file containing id and body columns
        sample_size (int): Number of samples to output
        output_file (str): File to write the results, if not specified prints to stdout
    """
    # Validate inputs
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    if not os.path.isfile(parquet_path):
        raise ValueError(f"Parquet file not found: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
        matching_ids = set()
        
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data['responses']:
                        matching_ids.add(item['id'])
        
        matching_bodies = df[df['id'].isin(matching_ids)]['body'].tolist()
        
        # Sample the bodies
        sampled_bodies = random.sample(matching_bodies, min(sample_size, len(matching_bodies)))
        
        # Output numbered list of bodies
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, body in enumerate(sampled_bodies, 1):
                    f.write(f"{i}. {strip_whitespace(body)}\n")
        else:
            for i, body in enumerate(sampled_bodies, 1):
                print(f"{i}. {strip_whitespace(body)}")
            
    except Exception as e:
        print(f"Error processing parquet file: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files and sample errors.")
    parser.add_argument("directory_path", type=str, help="The directory path containing JSON files.")
    parser.add_argument("parquet_path", type=str, help="The path to the parquet file.")
    parser.add_argument("--sample_size", type=int, default=20, help="The number of samples to output. Default is 20.")
    parser.add_argument("--output", type=str, help="The output file to write the results. If not specified, prints to stdout.")
    
    args = parser.parse_args()
    
    try:
        process_files(args.directory_path, args.parquet_path, args.sample_size, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
