
from chunking import split_into_chunks
from openai import OpenAI
from dotenv import load_dotenv
import os
import faiss
import numpy as np 

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load the flask documentation into chunks
chunks = split_into_chunks("docs/flask_basics.txt")

# print("Total chunks:",len(chunks))

# Generate an embedding for each chunk
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

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

user_query = "What is Flask's refund policy?"

# Convert users question into embeddings
query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_query
)

query_embedding = query_response.data[0].embedding

# Convert query embedding into Numpy array
query_vector = np.array([query_embedding],dtype="float32")

# Find to nearest vectors in faiss
distances,indices = index.search(query_vector,k=2)

# print("Distances:", distances)

# Check whether the nearest chunk is close enough
distance_threshold = 1.0

# Collect retrieved documentation chunks
relevant_chunks = []

for distance, chunk_index in zip(distances[0],indices[0]):
    if distance <= distance_threshold:
        relevant_chunks.append(chunks[chunk_index])

if not relevant_chunks:
    print("\nCopilot Answer:")
    print("I couldn't find sufficiently relevant information in the provided documentation.")
    exit()

# Use only the chunks that passed the threshold
context = "\n\n".join(relevant_chunks)

# print("\nRelevant chunks found:", len(relevant_chunks))
# print("\nFiltered Context:")
# print(context)

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
response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt
)

# Display the generated answer
print("Copilot Answer :")
print(response.output_text)