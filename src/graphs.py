"""
graphs.py

Script for processing JSON experiment results and generating plots.
Handles both inductive and online experiment results.
"""

import json
import os
from pathlib import Path
from src.functions.plots import Plots

# Placeholder for key mapping - replace with actual mapping
KEY_MAPPING_ONLINE = {
    "RR - CPDM v1": "LSPM v1",
    "RR - CPDM v2": "LSPM v2",
    "KNN - CPDM v1": "NNPM v1",
    "KNN - CPDM v2": "NNPM v2",
    "KRR - CPDM v1": "KRRPM v1",
    "KRR - CPDM v2": "KRRPM v2",
    "RR - PPDM": "RR",
    "KNN - PPDM": "KNN",
    "KRR - PPDM": "KRR",
    "GPR - BDT": "GPR",
    "BRR - BDT": "BRR",
    "Optimal": "Optimal",
    "NNPM v1": "NNPM v1",
    "NNPM v2": "NNPM v2",
    "LSPM v1": "LSPM v1",
    "LSPM v2": "LSPM v2",   
}

KEY_MAPPING_INDUCTIVE = {
    "RR - CPDM v2": "RR - CPDM",
    "KNN - CPDM v2": "KNN - CPDM",
    "KRR - CPDM v2": "KRR - CPDM",
    "RR - PPDM": "RR",
    "KNN - PPDM": "KNN",
    "KRR - PPDM": "KRR",
    "GPR - BDT": "GPR",
    "Optimal": "Optimal",
}

def determine_output_folder(filename):
    """Determine the output folder based on the filename."""
    if "f1" in filename:
        return os.path.join("data", "plots", "friedman1")
    elif "f2" in filename:
        return os.path.join("data", "plots", "friedman2")
    elif "f3" in filename:
        return os.path.join("data", "plots", "friedman3")
    elif "linear" in filename:
        return os.path.join("data", "plots", "linear")
    else:
        raise ValueError(f"Could not determine output folder for file: {filename}")

def process_json_file(file_path):
    """Process a single JSON file and generate appropriate plot."""
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get filename without extension for output
    filename = os.path.basename(file_path)
    
    # Choose the appropriate key mapping based on the filename
    if "inductive" in filename.lower():
        key_mapping = KEY_MAPPING_INDUCTIVE
        output_filename = os.path.splitext(filename)[0] + "_average_utility.pdf"
    else:
        key_mapping = KEY_MAPPING_ONLINE
        output_filename = os.path.splitext(filename)[0] + "_regret.pdf"
    
    # Apply key mapping
    mapped_data = {}
    for old_key, new_key in key_mapping.items():
        if old_key in data:
            mapped_data[new_key] = data[old_key]
        
    # Determine output folder
    output_folder = determine_output_folder(filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate appropriate plot based on filename
    if "inductive" in filename.lower():
        Plots.plot_average_utility(
            experiment=mapped_data,
            output_folder=output_folder,
            output_filename=output_filename
        )
    else:
        Plots.plot_cumulative_regret(
            experiment=mapped_data,
            output_folder=output_folder,
            output_filename=output_filename
        )

def main():
    """Main function to process all JSON files in the raw_data directory."""
    raw_data_dir = os.path.join("data", "raw_data")
    
    # Ensure the raw_data directory exists
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    # Process all JSON files in the directory
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(raw_data_dir, filename)
            try:
                process_json_file(file_path)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
