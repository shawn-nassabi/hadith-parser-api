import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict
from pydantic import BaseModel

LLM_MODEL="gpt-4.1-nano-2025-04-14"

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

class Hadith(BaseModel):
  sanad: list[str]         # Original Arabic chain
  sanad_en: list[str]      # English translation of chain
  matn: str          # Original Arabic matn
  matn_en: str       # English translation of matn


class AllHadith(BaseModel):
  hadiths: list[Hadith]

class HadithArabic(BaseModel):
  sanad: list[str]
  matn: str


class AllHadithArabic(BaseModel):
  hadiths: list[HadithArabic]

# Function that calls openai api to extract information from the given text file. Can specify whether english is needed or not
def extract_hadith_info(text: str, include_english: bool = True) -> AllHadith:
  """
  Extracts hadith info (sanad, matn) from text using OpenAI API, with English translation.
  Returns a Pydantic model (AllHadith).
  """

  system_prompt = (
    "You are an expert at structured data extraction from Islamic Ahadith text. "
    "You will be given unstructured text from various ahadith sources and need to extract all of them into the given structure."
    "When extracting the sanad, make sure to only include the sanad and not the matn or hadith text in the field for sanad. The sanad should only contain a list of the narrators (in order)"
    "Similarly, the field for matn should contain the main hadith text only. Not the sanad"
    "Please make sure you extract data for all of the hadiths appropriately. To stay consistent, use 'ibn' when writing the names in english, instead of bin."
  )

  system_prompt_ar = (
    "You are an expert at structured data extraction from Islamic Ahadith text. "
    "You will be given unstructured text in arabic from various ahadith sources and need to extract all of them into the given structure."
    "When extracting the sanad, make sure to only include the sanad and not the matn or hadith text in the field for sanad. The sanad should only contain a list of the narrators (in order)"
    "Similarly, the field for matn should contain the main hadith text only. Not the sanad"
    "Please make sure you extract data for all of the hadiths appropriately in the language given to you (usually arabic)."
  )
  
  # user_prompt = f"""Extract structured data for all ahadith in the following text:\n\n{text}\n\n"""
  user_prompt = f"""{text}"""
  if include_english:
    response = client.responses.parse(
      model=LLM_MODEL,
      input=[
        {
          "role": "system",
          "content": system_prompt
        },
        {"role": "user", "content": user_prompt},
      ],
      text_format=AllHadith,
    )
    # print(response)
    return response.output_parsed
  else:
    response = client.responses.parse(
      model=LLM_MODEL,
      input=[
        {
          "role": "system",
          "content": system_prompt_ar
        },
        {"role": "user", "content": user_prompt},
      ],
      text_format=AllHadithArabic,
    )
    print(response)
    return response.output_parsed

