import pandas as pd
import os

def prepare_data():
    """
    Reads all '_merged.csv' files from the 'data' directory, standardizes them,
    and combines them into a single CSV file for model training.
    """
    data_folder = 'data'
    output_filename = 'all_products.csv'
    
    if not os.path.exists(data_folder):
        print(f"Error: The '{data_folder}' directory was not found.")
        return

    # Look specifically for the new, richer data files
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('_merged.csv')]
    
    if not csv_files:
        print(f"Error: No '_merged.csv' files found in the '{data_folder}' directory.")
        return

    all_dfs = []
    
    print("Reading and processing new merged CSV files...")
    for file in csv_files:
        try:
            # Correctly extract category name from files like 'electronics_merged.csv'
            category = file.replace('_merged.csv', '').capitalize()
            filepath = os.path.join(data_folder, file)
            
            df = pd.read_csv(filepath)
            df['category'] = category
            
            # Standardize the review separator from '|' to '|||' for consistency
            if 'reviews' in df.columns:
                df['reviews'] = df['reviews'].str.replace('|', '|||', regex=False)

            all_dfs.append(df)
            print(f" - Successfully processed {file}")
        except Exception as e:
            print(f" - Could not process {file}. Error: {e}")

    if not all_dfs:
        print("No dataframes were created. Exiting.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.columns = combined_df.columns.str.strip().str.lower()
    combined_df.dropna(subset=['reviews', 'product_name'], inplace=True)
    
    output_filepath = os.path.join(data_folder, output_filename)
    combined_df.to_csv(output_filepath, index=False)
    
    print(f"\nSuccessfully combined {len(all_dfs)} files into '{output_filepath}'.")
    print(f"Total products processed with new data: {len(combined_df)}")

if __name__ == '__main__':
    prepare_data()

