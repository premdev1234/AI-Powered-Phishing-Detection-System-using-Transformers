import pandas as pd
import os

# Folder path containing the CSV files
folder_path = r"D:\mscis\dataset\24899952"

# List all CSV files in folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

merged_list = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)

    try:
        # ✅ Modern pandas way + avoid DtypeWarning
        df = pd.read_csv(
            file_path, encoding="utf-8", encoding_errors="ignore", low_memory=False
        )
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        continue

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Select only required columns
    cols_needed = ["label", "subject", "body", "sender", "receiver", "date", "urls"]
    available_cols = [col for col in cols_needed if col in df.columns]

    df = df[available_cols]

    # Add missing columns with NaN for consistency
    for col in cols_needed:
        if col not in df.columns:
            df[col] = pd.NA

    # Reorder columns
    df = df[cols_needed]

    merged_list.append(df)

# Concatenate all dataframes
if merged_list:
    final_df = pd.concat(merged_list, ignore_index=True)

    # Drop rows missing critical fields
    final_df.dropna(subset=["label", "subject", "body"], inplace=True)

    # Save merged CSV
    output_file = r"D:\mscis\dataset\consolidated_emails.csv"
    final_df.to_csv(output_file, index=False)

    print(f"✅ Merged dataset saved to {output_file} with {final_df.shape[0]} rows")
else:
    print("[INFO] No CSV files found or all failed to load.")
