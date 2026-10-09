
# Open the documentation file 
with open("docs/flask_basics.txt","r",encoding="utf-8") as file:
    text = file.read()

# Spilt the document into lines
lines = text.splitlines()

chunks = []
curr_chunk = []

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
            chunks.append("\n".join(curr_chunk))

        # start a new chunk with heading
        curr_chunk = [line]

    else:
        # add normal content to the current chunk
        curr_chunk.append(line)

# Add final chunk
if curr_chunk:
    chunks.append("\n".join(curr_chunk))

# Display the chunks 
for i,chunk in enumerate(chunks,start=1):
    print(f"\n----chunk{i}----")
    print(chunk)