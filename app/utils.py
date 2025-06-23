
MAX_LENGTH=1000000

# Check if the file is too large, and call chunk_text to split into chunks if needed
def parse_txt_file(text: str):
  if len(text) < MAX_LENGTH:
      return [text.strip()]
  else:
      return [h.strip() for h in chunk_text(text)]


# Function used to split large text files into multiple chunks (so that each chunk fits in the context window)
def chunk_text(hadith_text, max_chars = MAX_LENGTH):
  hadiths = hadith_text.strip().split("\n\n")  # split ahadith based on double new line delimiter
  chunks = []
  current_chunk = []
  current_length = 0
  for h in hadiths:
    # +2 for \n\n that gets stripped away in split
    if current_length + len(h) + 2 > max_chars and current_chunk:
      chunks.append('\n\n'.join(current_chunk))
      current_chunk = []
      current_length = 0
    current_chunk.append(h)
    current_length += len(h) + 2
  
  if current_chunk: # add the remaining chunk too (if there is any)
    chunks.append('\n\n'.join(current_chunk))
  return chunks