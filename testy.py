from google import genai
import os

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1"}
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say OK"
)

print(response.text)