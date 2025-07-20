import csv
import argparse
import os
import pandas as pd
import io
import requests
from openai import OpenAI

# Parse output filename from command-line
parser = argparse.ArgumentParser(description="Hadith API tester")
parser.add_argument(
    "--output",
    type=str,
    default="test_runs/hadith_api_comparison_output.txt",
    help="Output filename for results (default: hadith_api_comparison_output.txt)"
)
args = parser.parse_args()
filename = args.output
if not filename.startswith("test_runs/"):
    filename = os.path.join("test_runs", filename)
output_file = filename

openai_api_key = os.getenv("OPENAI_API_KEY") 

openai_client = OpenAI(api_key=openai_api_key)
from random import randrange
import sys

# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)

rand_seed = randrange(100)
if n >= 1:
	print("random seed = ", sys.argv[1])
	rand_seed = int(sys.argv[1])
result_number = ""
if n >= 2:
	print("result_number = ", sys.argv[2])
	result_number = int(sys.argv[2])

# Define the file paths for your two CSV files
hadith_path = "datasets/kaggle/kaggle_hadiths_clean.csv"
rawis_path = "datasets/kaggle/kaggle_rawis.csv"

#-------- Function to use LLM for checking if extracted sanad is valid in scenarios where the first check fails --------
def llm_name_match(extracted: str, true_names: list[str]) -> bool:
	print('Inside LLM check for the following:\n')

	print("Extracted sanad:", extracted)
	print("true-names:", true_names)

	prompt = (
		"You are an expert in Arabic names, specifically in the context of arabic Ahadith narrators (sanad). Compare the following extracted narrator name "
		"with the list of ground truth narrator names. Only answer 'YES' if the extracted name is a valid reference "
		"to any of the names in the list (e.g., partial, nickname, spelling, bin/ibn variations). It could also be that the person has another valid nickname or title, so please use your knowledge to judge accordingly. Only answer 'YES' or 'NO'.\n\n"
		f"Extracted name: {extracted}\n"
		f"Ground truth list: {true_names}\n\n"
		"Is the extracted name a valid match for any in the list? YES or NO:"
	)
	try:
		response = openai_client.responses.create(
			model="gpt-4o-mini-2024-07-18",
			input=prompt,
			temperature=0,
		)
		ans = response.output_text

		print('LLM check said:', ans)
		return "YES" in ans
	except Exception as e:
		print(f"[LLM ERROR] {e}")
		return False  # Fail safe: don't count as match

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
first_10_elements = hadith_df[['chain_indx', 'text_ar', 'source', 'chapter_no', 'hadith_no']].sample(n=sample_size,random_state=rand_seed)



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

with open(output_file, "w", encoding="utf-8") as f:
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
		unmatched_names = []
		for name in extracted_sanad:
			ex_tokens = name.split()  # e.g. ["ثابت", "البناني"]
			found = False
			for full_name in true_names:
				true_tokens = full_name.split()  # e.g. ["ثابت","بن","أسلم","البناني"]
				# check that every token in the extracted snippet appears somewhere
				if all(tok in true_tokens for tok in ex_tokens):
					matches += 1
					found = True
					break
			# Fuzzy fallback with LLM
			if not found:
				if llm_name_match(name, true_names):
					matches += 1
					found = True
			if not found:
					unmatched_names.append(name)
					

		# Metrics
		total_extracted += len(extracted_sanad)
		unique_true_names = list(set(name.strip() for name in true_names))
		total_ground_truth += len(unique_true_names)
		total_matched += matches
		if matches == len(unique_true_names) and len(extracted_sanad) == len(unique_true_names):
			exact_match_count += 1

		output = (
			f"\n🔹 Hadith: {key}\n"
			f"Extracted sanad: {extracted_sanad}\n"
			f"Ground truth:    {true_names}\n"
			f"✅ Matched {matches}/{len(extracted_sanad)} narrators\n"
		)
		
		if unmatched_names:
			output += f"❌ Unmatched extracted names: {unmatched_names}\n"

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