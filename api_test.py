#import requests

#response = requests.get("https://randomfox.ca/floof")

#print(response.status_code) #shows either 200 or 'OK'
#print(response.text) #shows things you can do
#print(response.json()) #shows actual things you can do with the 'dictionary'

#fox = response.json()
#print(fox['image'])

'''
base_url = "https://pokeapi.co/api/v2/"

def get_pokemon_info(name):
    url = f"{base_url}/pokemon/{name}"
    response = requests.get(url)

    if response.status_code == 200:
        pokemon_data = response.json()
        return pokemon_data

    else:
        print(f"Failed to retrieve data {response.status_code}")


pokemon_name = "typhlosion"
pokemon_info = get_pokemon_info(pokemon_name)

if pokemon_info:
    print(f"Name: {pokemon_info['name']}")
    print(f"ID: {pokemon_info['id']}")
    print(f"Height: {pokemon_info['height']}")
    print(f"Weight: {pokemon_info['weight']}")
'''

import google.generativeai as genai
def response_feedback(text):
    genai.configure(api_key="AIzaSyDvtnxy-N72AM0LDvbWoBiTK_Rw5CXjTec")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Explain how AI works")
    print(response.text)
    pass

genai.configure(api_key="AIzaSyDvtnxy-N72AM0LDvbWoBiTK_Rw5CXjTec")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("")
print(response.text)




"""
make it so that it inputs speetch and 
then outputs it in around 100 words


using deepspeech
"""

