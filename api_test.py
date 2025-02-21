from dotenv import load_dotenv
import os
import google.generativeai as genai
from whisper_test import main

word_caller = main()

load_dotenv(dotenv_path="API.env") 

API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(f"{word_caller}")
print(response.text)


"""
make it so that it inputs speetch and 
then outputs it in around 100 words

using Whisper
"""

