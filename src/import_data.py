import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/manual", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)
os.makedirs("src", exist_ok=True)

print("--- Step 2: Importing HuggingFace datasets ---")

# 1. Goan_Data
try:
    print("Loading konkani/Goan_Data...")
    ds_goan = load_dataset("konkani/Goan_Data")
    # Convert split to pandas
    if isinstance(ds_goan, dict) or hasattr(ds_goan, 'keys'):
        split_name = list(ds_goan.keys())[0]
        df_goan = ds_goan[split_name].to_pandas()
    else:
        df_goan = ds_goan.to_pandas()
    
    df_goan.to_csv("data/raw/goan_data.csv", index=False)
    print(f"Goan_Data shape: {df_goan.shape}")
    print("Goan_Data first 3 rows:")
    print(df_goan.head(3))
except Exception as e:
    print(f"Error loading konkani/Goan_Data: {e}")

# 2. Alpaca Konkani
try:
    print("\nLoading saillab/alpaca-konkani-cleaned...")
    ds_alpaca = load_dataset("saillab/alpaca-konkani-cleaned")
    if isinstance(ds_alpaca, dict) or hasattr(ds_alpaca, 'keys'):
        split_name = list(ds_alpaca.keys())[0]
        df_alpaca = ds_alpaca[split_name].to_pandas()
    else:
        df_alpaca = ds_alpaca.to_pandas()
    
    df_alpaca.to_csv("data/raw/alpaca_konkani.csv", index=False)
    print(f"Alpaca_Konkani shape: {df_alpaca.shape}")
    print("Alpaca_Konkani first 3 rows:")
    print(df_alpaca.head(3))
except Exception as e:
    print(f"Error loading saillab/alpaca-konkani-cleaned: {e}")
