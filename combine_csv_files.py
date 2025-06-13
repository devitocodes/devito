#!/usr/bin/env python3
"""
Script to combine multiple CSV files with similar patterns into unified files.
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def combine_csv_files_by_pattern(base_path, pattern, output_filename):
    """
    Combine CSV files matching a pattern into a single file.
    
    Parameters:
    - base_path: Base directory path
    - pattern: File pattern (e.g., 'conv_poly_coeff_s_*.csv')
    - output_filename: Name for the combined output file
    """
    
    # Find all files matching the pattern
    search_pattern = os.path.join(base_path, '**', pattern)
    files = glob.glob(search_pattern, recursive=True)
    files.sort()  # Sort to ensure consistent ordering
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} files matching pattern '{pattern}':")
    
    combined_data = {}
    
    for file_path in files:
        # Extract the number/order from filename
        filename = os.path.basename(file_path)
        # Extract number between 's_' and '.csv'
        try:
            order = int(filename.split('s_')[1].split('.csv')[0])
        except (IndexError, ValueError):
            print(f"Warning: Could not extract order from {filename}")
            continue
            
        print(f"  Processing {filename} (order {order})")
        
        try:
            # Read the CSV file
            if 'coeff' in pattern:
                # For coefficient files, handle complex numbers
                data = np.genfromtxt(file_path, delimiter=',', dtype=complex)
                combined_data[order] = data
            else:
                # For other files, read as regular CSV
                data = pd.read_csv(file_path, header=None)
                combined_data[order] = data.values.flatten()
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not combined_data:
        print(f"No valid data found for pattern: {pattern}")
        return
        
    # Create output directory if it doesn't exist
    output_dir = os.path.join(base_path, 'combined')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Save combined data
    if 'coeff' in pattern:
        # Save coefficient data as a structured format
        save_coefficient_data(combined_data, output_path)
    else:
        # Save other data as regular CSV
        save_regular_data(combined_data, output_path)
    
    print(f"Combined data saved to: {output_path}")

def save_coefficient_data(data_dict, output_path):
    """Save coefficient data in a structured format."""
    
    # Create a structured format: order, coefficient_index, real_part, imag_part
    rows = []
    
    for order in sorted(data_dict.keys()):
        coeffs = data_dict[order]
        for i, coeff in enumerate(coeffs):
            if np.isscalar(coeff):
                real_part = float(np.real(coeff))
                imag_part = float(np.imag(coeff))
            else:
                # Handle arrays
                real_part = float(np.real(coeff))
                imag_part = float(np.imag(coeff))
            
            rows.append([order, i, real_part, imag_part])
    
    df = pd.DataFrame(rows, columns=['order', 'coeff_index', 'real_part', 'imag_part'])
    df.to_csv(output_path, index=False)

def save_regular_data(data_dict, output_path):
    """Save regular numerical data."""
    
    rows = []
    for order in sorted(data_dict.keys()):
        values = data_dict[order]
        if np.isscalar(values):
            rows.append([order, values])
        else:
            for i, val in enumerate(values):
                rows.append([order, i, val])
    
    # Determine column names based on data structure
    if len(rows) > 0 and len(rows[0]) == 2:
        columns = ['order', 'value']
    else:
        columns = ['order', 'index', 'value']
        
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)

def main():
    """Main function to combine all CSV files by pattern."""
    
    base_path = '/home/fernan/Desktop/Trabajo/pos-doc/posdoc_fernando/code_snipets'
    
    # Define patterns and output filenames
    patterns_and_outputs = [
        ('conv_poly_coeff_s_*.csv', 'conv_poly_coefficients_combined.csv'),
        ('stab_poly_coeff_s_*.csv', 'stab_poly_coefficients_combined.csv'),
        ('conv_h_max_s_*.csv', 'conv_h_max_combined.csv'),
        ('stab_h_max_s_*.csv', 'stab_h_max_combined.csv'),
        ('poly_coeff_s_*.csv', 'poly_coefficients_combined.csv'),
    ]
    
    print("=" * 60)
    print("Combining CSV files by pattern")
    print("=" * 60)
    
    for pattern, output_name in patterns_and_outputs:
        print(f"\nProcessing pattern: {pattern}")
        print("-" * 40)
        combine_csv_files_by_pattern(base_path, pattern, output_name)
    
    print("\n" + "=" * 60)
    print("All files processed!")
    print("=" * 60)

if __name__ == "__main__":
    main()