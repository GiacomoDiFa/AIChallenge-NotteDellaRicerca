import json
import random

with open('texts.json','r') as file:
    data = json.load(file)

#print(data)

#for i in data['texts']:
    #print(i['text'])

testo = random.choice(list(data['texts']))

print("Testo casuale:",testo['text'])
print("Voce casuale:",testo['voice'])