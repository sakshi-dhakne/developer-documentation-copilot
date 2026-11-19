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


# Load documentation into chunks
def load_documents():
    doc_files = glob.glob("docs/*.txt")

    if not doc_files:
        print("No documentation files found in the docs folder.")
        return []

    chunks = []

    for file_path in doc_files:
        file_chunks = split_into_chunks(file_path)

        if not file_chunks:
            print(f"Warning: {file_path} is empty or contains no valid sections.")
            continue

        chunks.extend(file_chunks)

    if not chunks:
        print("No valid documentation content found.")
        return []

    print("Total documents:", len(doc_files))
    print("Total chunks:", len(chunks))

    return chunks


# Generate an embedding for each chunk
def create_embeddings(chunks):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[chunk['content'] for chunk in chunks]
        )

    except Exception as e:
        print("Error while generating embeddings:", e)
        return None

    embeddings = [item.embedding for item in response.data]

    embedding_array = np.array(embeddings, dtype="float32")

    return embedding_array


# Create FAISS index
def create_faiss_index(embedding_array):
    dimensions = embedding_array.shape[1]

    index = faiss.IndexFlatL2(dimensions)
    index.add(embedding_array)

    return index


# Retrieve relevant documentation chunks
def retrieve_chunks(user_query, index, chunks):
    # Convert user's question into embeddings
    try:
        query_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        )

    except Exception as e:
        print("Error while generating query embedding:", e)
        return None

    query_embedding = query_response.data[0].embedding

    query_vector = np.array([query_embedding], dtype="float32")

    # Search FAISS for nearest vectors
    distances, indices = index.search(query_vector, k=5)

    print("Distances:", distances)
    print("Retrieved chunk positions:", indices)

    # Apply distance threshold
    distance_threshold = 1.4
    relevant_chunks = []

    for distance, chunk_index in zip(distances[0], indices[0]):
        if chunk_index == -1:
            continue

        if distance <= distance_threshold:
            relevant_chunks.append(chunks[chunk_index])

    # Check relevance of retrieved chunks
    verified_chunks = []

    for chunk in relevant_chunks:
        label = check_relevance(user_query, chunk['content'])

        print(f"\nRelevance label: {label}")

        if label in ["RELEVANT", "PARTIALLY RELEVANT"]:
            verified_chunks.append(chunk)

    # Return only verified chunks
    return verified_chunks


# Generate answer using retrieved documentation
def generate_answer(user_query, relevant_chunks):

    # Prepare documentation context
    context = "\n\n".join(
        chunk['content'] for chunk in relevant_chunks
    )

    # Build prompt
    prompt = f"""
You are a Developer Documentation Copilot.

Answer the user's question using only the documentation provided below.
If the documentation does not contain the answer, clearly say that
the answer was not found in the provided documentation.
Do not invent information.

Documentation:
{context}

User question:
{user_query}

Answer:
"""

    # Send prompt to LLM
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

    except Exception as e:
        print("Error while generating the answer:", e)
        return None

    return response.output_text


# Main function
def main():

    # Load documents
    chunks = load_documents()

    if not chunks:
        return

    for chunk in chunks:
        print(f"\nSource: {chunk['source']}")
        print(f"Heading: {chunk['heading']}")

    # Generate embeddings
    embedding_array = create_embeddings(chunks)

    if embedding_array is None:
        return

    # Create FAISS index
    index = create_faiss_index(embedding_array)

    # Get user question
    user_query = "How do I define a Python function?"

    if not user_query.strip():
        print("Please enter a valid question.")
        return

    # Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(user_query, index, chunks)

    if relevant_chunks is None:
        return

    if not relevant_chunks:
        print("\nCopilot Answer:")
        print("I couldn't find sufficiently relevant information in the provided documentation.")
        return

    # Generate answer
    answer = generate_answer(user_query, relevant_chunks)

    if answer is None:
        return

    # Display answer
    print("\nCopilot Answer:")
    print(answer)

    # Display source references
    print("\nSource References:")

    for chunk in relevant_chunks:
        print(f"- {chunk['source']} | Section: {chunk['heading']}")


if __name__ == "__main__":
    main()
