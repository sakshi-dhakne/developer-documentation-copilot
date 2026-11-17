
from chunking import split_into_chunks
from openai import OpenAI
from dotenv import load_dotenv
import os
import faiss
import numpy as np 
from relevance_checker import check_relevance
import glob

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load the flask documentation into chunks
doc_files = glob.glob("docs/*.txt")

if not doc_files:
    print("No documentation files found in the docs folder.")
    exit()

chunks = []

for file_path in doc_files:
    file_chunks = split_into_chunks(file_path)

    if not file_chunks:
        print(f"Warning: {file_path} is empty or contains no valid sections.")
        continue

    chunks.extend(file_chunks)

if not chunks:
    print("No valid documentation content found.")
    exit()

print("Total documents:",len(doc_files))
print("Total chunks:",len(chunks))

for chunk in chunks:
    print(f"\nSource:{chunk['source']}")
    print(f"\nHeading:{chunk['heading']}")

# Generate an embedding for each chunk
try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[chunk['content'] for chunk in chunks]
    )

except Exception as e:
    print("Error while generating embeddings:", e)
    exit()

# Extract the vectors
embeddings = [item.embedding for item in response.data]

# print("\nTotal Embeddings:",len(embeddings))
# print("Dimentions per enbedding:",len(embeddings[0]))

# Convert embedding into a Numpy array 
embedding_array = np.array(embeddings,dtype="float32")

# print("Embedding array shape :",embedding_array.shape)

# Store embedding in faiss
dimensions = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimensions)
index.add(embedding_array)

# print("vectors stored in faiss :",index.ntotal)

# Search for relevant documentation 

user_query = "How do I define a Python function?"

if not user_query.strip():
    print("Please enter a valid question.")
    exit()

# Convert users question into embeddings
query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_query
)

query_embedding = query_response.data[0].embedding

# Convert query embedding into Numpy array
query_vector = np.array([query_embedding],dtype="float32")

# Find to nearest vectors in faiss
distances,indices = index.search(query_vector,k=5)

print("Distances:", distances)
print("Retrieved chunk position:",indices)

print("\nRetrieved Chunk:")
for chunk_ind in indices[0]:

    if chunk_ind == -1:
        continue

    print(f"\nPosition:{chunk_ind}")

    chunk = chunks[chunk_ind]

    print("Source :",chunk['source'])
    print("Heading :",chunk['heading'])
    print("Content :",chunk['content'])

# Check whether the nearest chunk is close enough
distance_threshold = 1.4

# Collect retrieved documentation chunks
relevant_chunks = []

for distance, chunk_index in zip(distances[0],indices[0]):

    if chunk_index == -1:
        continue

    if distance <= distance_threshold:
        relevant_chunks.append(chunks[chunk_index])

# Check each retrieved chunk for relevance
verified_chunks = []
for chunk in relevant_chunks:
    label = check_relevance(user_query, chunk['content'])

    print(f"\nRelevance label:{label}")

    if label in ["RELEVANT","PARTIALLY RELEVANT"]:
        verified_chunks.append(chunk)

# Keep only verified chunks
relevant_chunks = verified_chunks

if not relevant_chunks:
    print("\nCopilot Answer:")
    print("I couldn't find sufficiently relevant information in the provided documentation.")
    exit()

# Use only the chunks that passed the threshold
context = "\n\n".join(
    chunk['content'] for chunk in relevant_chunks
    )

# Build the prompt using instructions, context and user question
prompt = f"""
You are a Developer Documentation Copilot.

Answer the user's question using only the documentation provided below.
If the documentation does not contain the answer, clearly say that
the answer was not found in the provided documentation.
Do not invent information.
     
Documentation :
{context}

User question :
{user_query}
    
Answer :
"""

# print("\nPrompt created successfully")
# print(prompt)

# Send the prompt to LLM
try:
  response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt
  )
except Exception as e:
    print("Error while generating the answer:", e)
    exit()

# Display the generated answer
print("\nCopilot Answer :")
print(response.output_text)

# Display source references
print("\nSource References :")

for chunk in relevant_chunks:
    print(f"-{chunk['source']} | Section:{chunk['heading']}")