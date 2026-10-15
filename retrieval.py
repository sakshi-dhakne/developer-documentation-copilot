
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

print("Total chunks:",len(chunks))

# # Display the chunks 
# for i,chunk in enumerate(chunks,start=1):
#     print(f"\n----chunk{i}----")
#     print(chunk)

# Generate an embedding for each chunk
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

# Extract the vectors
embeddings = [item.embedding for item in response.data]

print("\nTotal Embeddings:",len(embeddings))
print("Dimentions per enbedding:",len(embeddings[0]))

# Convert embedding into a Numpy array 
embedding_array = np.array(embeddings,dtype="float32")

print("Embedding array shape :",embedding_array.shape)

# Store embedding in faiss
dimensions = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimensions)
index.add(embedding_array)

print("vectors stored in faiss :",index.ntotal)


# Search for relevant documentation 

user_query = "How to access query parameter in flask"

# convert users question into embeddings
query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_query
)

query_embedding = query_response.data[0].embedding

# convert query embedding into Numpy array
query_vector = np.array([query_embedding],dtype="float32")

# Find to nearest vectors in faiss
distances,indices = index.search(query_vector,k=2)

# display retieve documentation chunks
for i,chunk_index in enumerate(indices[0]):
    print(f"\nResult{i+1}")
    print("Distance :",distances[0][i])
    print(chunks[chunk_index])