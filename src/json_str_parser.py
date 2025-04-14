import json
import os
import re
from google import genai


MODEL = 'gemini-2.0-flash'
API_KEY = os.environ['GEMINI_API_KEY']
client = genai.Client(api_key=API_KEY)

def parse(json_str, e):
    print('Using Gemini to fix json str')
    prompt = f'''Rewrite this string fix this error: {e}

    Here is the string I'm attempting to turn into a json with json.loads: {json_str}
    Only return the adjust string nothing else'''

    response = client.models.generate_content(
        model=MODEL, contents=prompt
        )
    
    json_str = response.text
    return json_str
