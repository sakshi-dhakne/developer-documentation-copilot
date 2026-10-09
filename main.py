
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

user_query = input("Ask your question : ")
client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-4o-mini",
    input=user_query
)

print(response.output_text)