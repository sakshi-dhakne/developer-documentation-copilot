import os 

def split_into_chunks(file_path):
  # Open the documentation file 
  with open(file_path,"r",encoding="utf-8") as file:
    text = file.read()

  # Get the documentation file name 
  source = os.path.basename(file_path)

  # Spilt the document into lines
  lines = text.splitlines()

  chunks = [] 
  curr_chunk = []
  current_heading = None

  # Go through each line 
  for line in lines:
    line = line.rstrip()

    # Skip completely empty lines
    if not line:
      continue

    # Check if line looks like a heading
    if line.isupper():
      # saving previous chunk before starting a new one
      if curr_chunk:
        chunks.append({
          'source':source,
          'heading':current_heading,
          'content':"\n".join(curr_chunk)
        })

      # start a new chunk with heading
      current_heading = line
      curr_chunk = [line]

    else:
      # add normal content to the current chunk
      curr_chunk.append(line)

  # Add final chunk 
  if curr_chunk:
    chunks.append({
      'source':source,
      'heading':current_heading,
      'content':"\n".join(curr_chunk)
    })

  return chunks