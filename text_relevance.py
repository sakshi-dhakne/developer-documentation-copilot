
from dotenv import load_dotenv
from relevance_checker import check_relevance

load_dotenv()

user_query =  "How can I create a Flask route that accepts JSON data?"

chunk = """
REQUEST DATA

Use request.get_json() to read JSON data
sent in a POST request.

Example:

data = request.get_json()
name = data.get("name")
"""

label = check_relevance(user_query, chunk)

print("Relevance label:", label)
