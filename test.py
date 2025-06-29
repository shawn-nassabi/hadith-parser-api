import csv
import os
import pandas as pd

# Define the file paths for your two CSV files
hadith_path = "datasets/kaggle/kaggle_hadiths_clean.csv"
rawis_path = "datasets/kaggle/kaggle_rawis.csv"

try:
	hadith_df = pd.read_csv(hadith_path)
	print(f"DataFrame 1 loaded successfully from {hadith_path}")
	print("First 5 rows of hadith_df:")
	print(hadith_df.head())
except FileNotFoundError:
	print(f"Error: {hadith_path} not found. Please ensure the file exists and the path is correct.")
except Exception as e:
	print(f"An error occurred while reading {hadith_path}: {e}")

try:
	rawis_df = pd.read_csv(rawis_path)
	print(f"\nDataFrame 2 loaded successfully from {rawis_path}")
	print("First 5 rows of rawis_df:")
	print(rawis_df.head())
except FileNotFoundError:
	print(f"Error: {rawis_path} not found. Please ensure the file exists and the path is correct.")
except Exception as e:
	print(f"An error occurred while reading {rawis_path}: {e}")

sample_size = 10
first_10_elements = hadith_df[['chain_indx', 'text_ar']].sample(n=sample_size)


chain_indices_list = first_10_elements['chain_indx'].to_numpy().tolist()
matn_and_sand_list = first_10_elements['text_ar'].to_numpy().tolist()
pd.set_option('display.max_columns', None)
for chains,arabic_text in zip(chain_indices_list,matn_and_sand_list):
	print(arabic_text)
	chain_list = chains.split(",")
	for narr_id_num,count in zip(chain_list,range(1,len(chain_list))):
		result = rawis_df[rawis_df['scholar_indx'] == int(narr_id_num)]
		# it needs to be a numpy array to get the full name, pandas series or str won't work
		numpy_array = result['name'].to_numpy()
		print(count, numpy_array)
	