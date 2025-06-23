from fastapi import APIRouter, UploadFile, File, Query
from app.utils import parse_txt_file
from app.openai_client import extract_hadith_info

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...), include_english: bool = Query(
      True,
      description="Whether to include English translations (sanad_en, matn_en)."
    )):
    content = await file.read()
    text = content.decode("utf-8")
    
    text_chunks = parse_txt_file(text)  # Split into individual hadiths (you'll define this)
    
    results = []
    for chunk in text_chunks:
        structured_data = extract_hadith_info(chunk, include_english=include_english)
        results.append(structured_data)
    
    return {"results": results}