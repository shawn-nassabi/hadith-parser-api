import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAIError  # or the specific exception your client throws
import json
# gpt-4o-2024-08-06
# gpt-4o-mini-2024-07-18
# gpt-4.1-2025-04-14
# gpt-4.1-nano-2025-04-14
LLM_MODEL="gpt-4o-mini-2024-07-18"

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# To extract sanad and matn in both english and arabic
class Hadith(BaseModel):
  sanad: list[str]         # Original Arabic chain
  sanad_en: list[str]      # English translation of chain
  matn: str          # Original Arabic matn
  matn_en: str       # English translation of matn


class AllHadith(BaseModel):
  hadiths: list[Hadith]

# To extract only arabic sanad and matn
class HadithArabic(BaseModel):
  sanad: list[str]
  matn: str


class AllHadithArabic(BaseModel):
  hadiths: list[HadithArabic]


# To extract only sanad and the sanad sentence to get the harf
class HadithSanadWithEnglish(BaseModel):
  source: str
  chapter_no: int
  hadith_no: int
  sanad: list[str]
  sanad_sentence: str
  sanad_english: list[str]
  sanad_sentence_english: str

class AllHadithSanadWithEnglish(BaseModel):
  hadiths: list[HadithSanadWithEnglish]


class HadithSanad(BaseModel):
  source: str
  chapter_no: int
  hadith_no: int
  sanad: list[str]
  sanad_sentence: str

class AllHadithSanad(BaseModel):
  hadiths: list[HadithSanad]



# Function that calls openai api to extract information from the given text file. Can specify whether english is needed or not
def extract_hadith_info_from_txt_file(text: str, include_english: bool = True) -> AllHadith:
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
    "You are an expert at structured data extraction from Islamic Ahadith text."
    "You will be given unstructured text in arabic from various ahadith sources and need to extract all of them into the given structure."
    "When extracting the sanad, make sure to only include the sanad and not the matn or hadith text in the field for sanad. The sanad should only contain a list of the narrators (in order)"
    "Similarly, the field for matn should contain the main hadith text only. Not the sanad"
    "Please make sure you extract data for all of the hadiths appropriately in the language given to you (usually arabic)."
  )
  
  # user_prompt = f"""Extract structured data for all ahadith in the following text:\n\n{text}\n\n"""
  user_prompt = f"""{text}"""

  system = system_prompt if include_english else system_prompt_ar
  fmt = AllHadith if include_english else AllHadithArabic

  response = client.responses.parse(
      model=LLM_MODEL,
      input=[
          {"role": "system", "content": system},
          {"role": "user", "content": user_prompt},
      ],
      text_format=fmt,
  )

  print(response)

  return response.output_parsed



# Retry up to 3 times with exponential backoff (1s, 2s, 4s)
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(OpenAIError)
)
def _call_openai_with_retry(system_prompt: str, user_prompt: str, with_english: bool) -> AllHadithSanad:
    # Choose output format depending on whether english is included or not
    fmt = AllHadithSanadWithEnglish if with_english else AllHadithSanad
    response = client.responses.parse(
      model=LLM_MODEL,
      input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ],
      text_format=fmt,
      temperature=1
    )

    return response.output_parsed

# For extracting the sanad
def extract_sanad_batch(hadith_entries: list[dict], with_english: bool = False) -> list[HadithSanad]:
    
    print("Inside extract_sanad_batch in openai_client.py file.")
    print("Extracting data from: \n")
    print(hadith_entries)
    
    if with_english:
      system_prompt = (
          "You are an expert in Islamic Hadith sciences. "
          "You will be given multiple hadith entries with metadata and Arabic text. "
          "For each entry, extract into the JSON the following fields:\n"
          "- source: the hadith source name\n"
          "- chapter_no: the chapter number\n"
          "- hadith_no: the hadith number\n"
          "- sanad: list of narrators (Arabic) in order.\n"
          "- sanad_sentence: the exact sentence containing the sanad. For example: حدثنا الحميدي عبد الله بن الزبير، قال حدثنا سفيان، قال حدثنا يحيى بن سعيد الأنصاري، قال أخبرني محمد بن إبراهيم التيمي، أنه سمع علقمة بن وقاص الليثي، يقول سمعت عمر بن الخطاب  رضى الل\n"
          "- sanad_english: list of narrators translated into English (use 'ibn' not 'bin')\n"
          "- sanad_sentence_english: the translated sanad sentence\n\n"
          "Here are a few examples of input and expected valid output:" 
          "START OF EXAMPLE 1:" 
          "### Input:" 
          "Source: Sahih Bukhari" 
          "Chapter: 1" 
          "Hadith No: 1" 
          "Text: حدثنا الحميدي عبد الله بن الزبير، قال حدثنا سفيان، قال حدثنا يحيى بن سعيد الأنصاري، قال أخبرني محمد بن إبراهيم التيمي، أنه سمع علقمة بن وقاص الليثي، يقول سمعت عمر بن الخطاب  رضى الله عنه  على المنبر قال سمعت رسول الله صلى الله عليه وسلم يقول إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى، فمن كانت هجرته إلى دنيا يصيبها أو إلى امرأة ينكحها فهجرته إلى ما هاجر إليه"
          "### Expected Output:"
          "source: Sahih Bukhari"
          "chapter_no: 1"
          "hadith_no: 1"
          "sanad: [\"عبد الله بن الزبير\", \"سفيان بن عيينة\", \"يحيى بن سعيد الأنصاري\", \"محمد بن إبراهيم بن الحارث\", \"علقمة بن وقاص\", \"عمر بن الخطاب\"]"
          "sanad_sentence: حدثنا الحميدي عبد الله بن الزبير، قال حدثنا سفيان، قال حدثنا يحيى بن سعيد الأنصاري، قال أخبرني محمد بن إبراهيم التيمي، أنه سمع علقمة بن وقاص الليثي، يقول سمعت عمر بن الخطاب"
          "sanad_english: [\"Umar ibn Al-Khattab\", \"Alqammah ibn Waqqas\", \"Muhammad ibn Ibrahim ibn Harith\", \"Yahya ibn Saeed Al-Ansari\", \"Sufyan ibn Aina\", \"Andullah ibn Zubayr\"]"
          "sanad_sentence_english: \"Al-Humaydi ‘Abdullah ibn al-Zubayr narrated to us, he said: Sufyan narrated to us, he said: Yahya ibn Sa‘id al-Ansari narrated to us, he said: Muhammad ibn Ibrahim al-Taymi informed me, that he heard ‘Alqamah ibn Waqqas al-Laythi say: I heard ‘Umar ibn al-Khattab\""
          "END OF EXAMPLE 1"
          "Extract the data from the ahadith into the given structure."
      )
    else:
       system_prompt = (
          "You are an expert in Islamic Hadith sciences."
          "You will be given multiple hadith entries with metadata and Arabic text."
          "For each entry, extract into the JSON the following fields:\n"
          "- source: the hadith source name\n"
          "- chapter_no: the chapter number\n"
          "- hadith_no: the hadith number\n"
          "- sanad: list of narrators (Arabic) in order.\n"
          "- sanad_sentence: the exact sentence containing the sanad. For example: حدثنا الحميدي عبد الله بن الزبير، قال حدثنا سفيان، قال حدثنا يحيى بن سعيد الأنصاري، قال أخبرني محمد بن إبراهيم التيمي، أنه سمع علقمة بن وقاص الليثي، يقول سمعت عمر بن الخطاب  رضى الل\n"
          "Here are a few examples of input and expected valid output:" 
          "START OF EXAMPLE 1:" 
          "### Input:" 
          "Source: Sahih Bukhari" 
          "Chapter: 1" 
          "Hadith No: 1" 
          "Text: حدثنا الحميدي عبد الله بن الزبير، قال حدثنا سفيان، قال حدثنا يحيى بن سعيد الأنصاري، قال أخبرني محمد بن إبراهيم التيمي، أنه سمع علقمة بن وقاص الليثي، يقول سمعت عمر بن الخطاب  رضى الله عنه  على المنبر قال سمعت رسول الله صلى الله عليه وسلم يقول إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى، فمن كانت هجرته إلى دنيا يصيبها أو إلى امرأة ينكحها فهجرته إلى ما هاجر إليه"
          "### Expected Output:"
          "source: Sahih Bukhari"
          "chapter_no: 1"
          "hadith_no: 1"
          "sanad: [\"عبد الله بن الزبير\", \"سفيان بن عيينة\", \"يحيى بن سعيد الأنصاري\", \"محمد بن إبراهيم بن الحارث\", \"علقمة بن وقاص\", \"عمر بن الخطاب\"]"
          "sanad_sentence: حدثنا الحميدي عبد الله بن الزبير، قال حدثنا سفيان، قال حدثنا يحيى بن سعيد الأنصاري، قال أخبرني محمد بن إبراهيم التيمي، أنه سمع علقمة بن وقاص الليثي، يقول سمعت عمر بن الخطاب"
          "END OF EXAMPLE 1"
          "Extract the data from the ahadith into the given structure."
        )

    # Build a single prompt by joining each entry’s metadata + text.
    prompt_parts = []
    for e in hadith_entries:
      prompt_parts.append(
        f"Source: {e['source']}\n"
        f"Chapter: {e['chapter_no']}\n"
        f"Hadith No: {e['hadith_no']}\n"
        f"Text: {e['text_ar']}"
      )

    user_prompt = "\n\n---\n\n".join(prompt_parts)

    # print(user_prompt)
    # print("\n\n")
    try:
      parsed_response = _call_openai_with_retry(system_prompt, user_prompt, with_english)
      return parsed_response.hadiths
    except OpenAIError as e:
      print("❌ OpenAI API failed after retries:", e)
      raise e