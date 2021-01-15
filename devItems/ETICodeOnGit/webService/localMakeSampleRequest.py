# importing the requests library
import requests
import json
import time

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
     'promptId': '12'},
    {'testId': '7',
     'content': 'Hola 5',
     'form': 'B',
     'level': 'A',
     'promptId': '11'},
    {'testId': '8',
     'content': 'Hola 6',
     'form': 'B',
     'level': 'I',
     'promptId': '12'}
]

# for i in range(0,10000):
#     item=responses[i%4]
#     responses.append(item)


strResponse = json.dumps(responses)

# defining a params dict for the parameters to be sent to the API
PARAMS = {'jsonData': strResponse}
start_time = time.time()
# sending get request and saving the response as response object
r = requests.get(url=URL, params=PARAMS)
print(type(r))
print(r.text)

# r = requests.post(url=URL, params=PARAMS)
# print(type(r))
# print(r.text)

dura=time.time() - start_time

print('Finish receive message in {} second'.format(dura))