import csv
import os
import pandas as pd
import io
import requests

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
first_10_elements = hadith_df[['chain_indx', 'text_ar', 'source', 'chapter_no', 'hadith_no']].sample(n=sample_size)



# Prepare the sample dataframe as a CSV in memory
csv_buffer = io.StringIO()
first_10_elements.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

# Send the CSV as a file to the FastAPI endpoint
response = requests.post(
    "http://localhost:8000/upload-csv/",
    files={"file": ("sample.csv", csv_buffer, "text/csv")}
)

# Handle and print the API response
if response.ok:
    results = response.json().get("results", [])
    print("\nAPI Response:")
    for r in results:
        print(r)
else:
    print("API request failed with status code:", response.status_code)
    print(response.text)



# lookup for (source, chapter_no, hadith_no) → chain_indx
chain_lookup = {
	(row['source'].strip(), row['chapter_no'], int(row['hadith_no'].strip())): row['chain_indx']
	for _, row in first_10_elements.iterrows()
}

print(chain_lookup)

# Cache to store rawis once found
rawi_cache = {}

def get_rawi_name(scholar_id: int) -> str:
    if scholar_id in rawi_cache:
        return rawi_cache[scholar_id]
    
    result = rawis_df[rawis_df['scholar_indx'] == scholar_id]
    if not result.empty:
        name = result.iloc[0]['name']
        rawi_cache[scholar_id] = name
        return name
    else:
        return ""
		

# Metric accumulators
total_extracted = 0
total_ground_truth = 0
total_matched = 0
exact_match_count = 0

with open("hadith_api_comparison_output.txt", "w", encoding="utf-8") as f:
	print("\n=== Comparison Results ===")
	f.write("\n=== Comparison Results ===\n")
	
	for api_result in results:
		key = (api_result['source'], api_result['chapter_no'], api_result['hadith_no'])
		chain_str = chain_lookup.get(key)

		if not chain_str:
			msg = f"❌ Chain not found for: {key}"
			print(msg)
			f.write(msg + "\n")
			continue

		narrator_ids = [int(i) for i in chain_str.split(',')]
		true_names = [get_rawi_name(i) for i in narrator_ids]
		extracted_sanad = api_result.get("sanad", [])

		matches = 0
		for name in extracted_sanad:
			if any(name in full_name for full_name in true_names):
				matches += 1

		# Metrics
		total_extracted += len(extracted_sanad)
		total_ground_truth += len(true_names)
		total_matched += matches
		if matches == len(true_names) and len(extracted_sanad) == len(true_names):
			exact_match_count += 1

		output = (
			f"\n🔹 Hadith: {key}\n"
			f"Extracted sanad: {extracted_sanad}\n"
			f"Ground truth:    {true_names}\n"
			f"✅ Matched {matches}/{len(extracted_sanad)} narrators\n"
		)
		print(output)
		f.write(output)

	# Final metrics
	precision = total_matched / total_extracted if total_extracted else 0
	recall = total_matched / total_ground_truth if total_ground_truth else 0
	f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
	exact_match_rate = exact_match_count / len(results) if results else 0

	summary = (
		"\n=== Evaluation Metrics ===\n"
		f"🔸 Precision:         {precision:.2f}\n"
		f"🔸 Recall:            {recall:.2f}\n"
		f"🔸 F1 Score:          {f1_score:.2f}\n"
		f"🔸 Exact Match Rate:  {exact_match_rate:.2f} ({exact_match_count}/{len(results)})\n"
	)
	print(summary)
	f.write(summary)