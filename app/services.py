import pandas as pd
from io import StringIO
from app.utils import parse_txt_file
from app.openai_client import extract_hadith_info_from_txt_file, extract_sanad_batch
from fastapi import UploadFile


# -------------------- PROCESS TXT UPLOAD ----------------------
async def process_txt_upload(file: UploadFile, include_english: bool = True):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = parse_txt_file(text)

    results = []
    for chunk in chunks:
        structured = extract_hadith_info_from_txt_file(chunk, include_english=include_english)
        results.append(structured)
    return results


# -------------------- PROCESS CSV UPLOAD ----------------------
async def process_csv_upload(file: UploadFile, include_english: bool = False):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    required_columns = ("source", "chapter_no", "hadith_no", "text_ar")
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")

    entries = [
        {
            "source": str(row["source"]),
            "chapter_no": int(row["chapter_no"]),
            "hadith_no": int(row["hadith_no"]),
            "text_ar": str(row["text_ar"]),
        }
        for _, row in df.iterrows()
    ]

    batch_size = 3
    all_results: list[dict] = []

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        try:
            hadiths_output = extract_sanad_batch(batch, with_english=include_english)
            all_results.extend(hadiths_output)
        except Exception as e:
            for j in range(len(batch)):
                all_results.append({"error": str(e), **batch[j]})

    return all_results