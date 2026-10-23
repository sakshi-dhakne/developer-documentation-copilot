from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. Check your .env file."
    )

client = OpenAI(api_key=api_key)


def check_relevance(user_query, chunk):
    prompt = f"""
Classify the chunk based on whether it helps answer ANY
meaningful part of the user's question.

Follow these rules in order:

1. RELEVANT:
   The chunk provides information that directly answers
   the user's question sufficiently.

2. PARTIALLY RELEVANT:
   The chunk answers at least one meaningful subtask,
   but does not provide enough information to answer
   the entire question.

3. IRRELEVANT:
   The chunk does not help answer any meaningful part
   of the user's question. Sharing the same topic or
   keywords is not sufficient.

Important example:

Question:
"How can I create a Flask route that accepts JSON data?"

Chunk:
"A route connects a URL to a Python function.
The @app.route() decorator defines a route."

Correct label: PARTIALLY RELEVANT

Reason:
The chunk explains route creation, which is one part
of the question, but does not explain JSON handling.

Do not label a chunk IRRELEVANT merely because it
does not answer the entire question.
When a chunk answers part, but not all, of the
question, choose PARTIALLY RELEVANT.


User question:
{user_query}

Documentation chunk:
{chunk}

Return only one label:
RELEVANT, PARTIALLY RELEVANT, or IRRELEVANT.
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    label = response.output_text.strip().upper()

    allowed_label = [
        "RELEVANT",
        "PARTIALLY RELEVANT",
        "IRRELEVANT"
    ]

    if label not in allowed_label:
        raise ValueError(f"Unexpected Relevance Label:{label}")

    return label