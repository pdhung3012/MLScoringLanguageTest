# importing the requests library
import requests
import json

# api-endpoint
URL = "http://127.0.0.1:5000/responses"

responses = [
    {'testId': '5',
     'content': 'Hola 5',
     'form': 'A',
     'level': 'I',
     'promptId': '11'},
    {'testId': '6',
     'content': 'Hola 6',
     'form': 'A',
     'level': 'N',
     'promptId': '12'}
]

strResponse = json.dumps(responses)

# defining a params dict for the parameters to be sent to the API
PARAMS = {'jsonData': strResponse}

# sending get request and saving the response as response object
r = requests.get(url=URL, params=PARAMS)

# extracting data in json format
# data = r.json()

print(type(r))
print(r.text)