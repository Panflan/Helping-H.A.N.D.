from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("API_KEY")
def response_feedback(text):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("What is 2+2 equal to?")
    print(response.text)
    pass
"""
make it so that it inputs speetch and 
then outputs it in around 100 words

using Whisper
"""

