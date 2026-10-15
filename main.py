
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# text we want to convert into vector
user_query = "Flask uses request.args to access query parameters."


response = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_query
)

# Get the vector from the response
embedding = response.data[0].embedding

# some info about vector 
print('embedding successful')
print('vector dimension',len(embedding))
print('First 10 values :',embedding[:10])