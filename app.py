from flask import Flask,request,jsonify

from retrieval import ( load_documents, create_embeddings, create_faiss_index, retrieve_chunks, generate_answer)

# Create Flask application
app = Flask(__name__)


# Initialize RAG pipeline
chunks = load_documents()

embedding_array = create_embeddings(chunks)

index = create_faiss_index(embedding_array)


@app.route("/")
def home():
    return "Developer Documentation Copilot API is running!"


# Question endpoint
@app.route("/ask", methods=["POST"])
def ask_question():

    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Please send valid JSON data."
        }), 400

    user_query = data.get("question")

    if not user_query or not user_query.strip():
        return jsonify({
            "error": "Question cannot be empty."
        }), 400

    try:    
        # Retrieve relevant document chunks
        relevant_chunks = retrieve_chunks(user_query, index, chunks)
        
        # Generate answer using LLM
        answer = generate_answer(user_query, relevant_chunks)

    except Exception as e:
        print("RAG Error:", e)

        return jsonify({
           "error": "An error occurred while processing your question."
        }), 500

    # Extract source references
    sources = [
       {
        "document": chunk["source"],
        "section": chunk["heading"]
       }
        for chunk in relevant_chunks
    ]

    # Return response
    return jsonify({
       "question": user_query,
       "answer": answer,
       "sources": sources
    }), 200


# Run the Flask application
if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)