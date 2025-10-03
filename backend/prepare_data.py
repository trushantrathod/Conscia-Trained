import pandas as pd
import os

def prepare_data():
    """
    Reads all source CSV files from the 'data' directory, adds a 'category' column based on the filename,
    cleans the data, and combines everything into a single 'all_products.csv' file.
    """
    # Get the absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the data folder relative to the script's location
    data_folder = os.path.join(script_dir, 'data')
    output_filename = 'all_products.csv'
    output_filepath = os.path.join(data_folder, output_filename)

    # --- 1. Validation ---
    if not os.path.exists(data_folder):
        print(f"Error: The '{data_folder}' directory was not found.")
        print("Please ensure your CSV files are inside a 'data' folder next to this script.")
        return

    # Find all CSV files in the data directory, excluding the output file itself
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and f != output_filename]
    
    if not csv_files:
        print(f"Error: No source CSV files found in the '{data_folder}' directory.")
        return

    # --- 2. Read and Process Files ---
    all_dfs = []
    print("Reading and processing CSV files...")
    for file in csv_files:
        try:
            # Extract a clean category name from the filename (e.g., 'electronics_merged.csv' -> 'Electronics')
            category = file.split('_')[0].capitalize()
            filepath = os.path.join(data_folder, file)
            
            df = pd.read_csv(filepath)
            df['category'] = category
            
            all_dfs.append(df)
            print(f" - Successfully processed '{file}' -> Category: '{category}'")
        except Exception as e:
            print(f" - Could not process {file}. Error: {e}")

    if not all_dfs:
        print("No data could be processed. Exiting.")
        return

    # --- 3. Combine and Clean Data ---
    print("\nCombining and cleaning data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Standardize column names (e.g., remove leading/trailing spaces, make lowercase)
    combined_df.columns = combined_df.columns.str.strip().str.lower()

    # Ensure crucial columns for the app exist
    required_cols = ['reviews', 'product_name', 'product_id']
    for col in required_cols:
        if col not in combined_df.columns:
            print(f"Error: Missing required column '{col}' in the combined data. Exiting.")
            return
            
    # Drop rows where crucial data is missing
    combined_df.dropna(subset=required_cols, inplace=True)
    
    # --- 4. Save the Result ---
    combined_df.to_csv(output_filepath, index=False)
    
    print("\nData preparation complete! ✨")
    print(f"Combined data from {len(all_dfs)} files into '{output_filename}'.")
    print(f"Total products ready for the app: {len(combined_df)}")

if __name__ == '__main__':
    prepare_data()