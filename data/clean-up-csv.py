#!/bin/env python3
import sys
import pandas as pd

# Check if correct number of arguments provided
if len(sys.argv) < 3:
    print("Usage: python script_name.py input_file.csv output_file.csv")
    sys.exit(1)

# Extract filenames from command-line arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# Read the CSV file
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print("Error: Input file not found.")
    sys.exit(1)

# Drop rows with missing values in any of the specified columns
df_filtered = df.dropna(subset=['observed_on', 'species_guess', 'latitude', 'longitude'])

# Keep only the desired columns
df_filtered = df_filtered[['observed_on', 'species_guess', 'latitude', 'longitude']]

# Save the filtered dataframe to a new CSV file
df_filtered.to_csv(output_file, index=False)
