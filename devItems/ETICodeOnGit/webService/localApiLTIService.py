import flask
from flask import request, jsonify
from base64 import b64encode,b64decode
import json
import traceback

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# # Create some test data for our catalog in the form of a list of dictionaries.
# books = [
#     {'id': 0,
#      'title': 'A Fire Upon the Deep',
#      'author': 'Vernor Vinge',
#      'first_sentence': 'The coldsleep itself was dreamless.',
#      'year_published': '1992'},
#     {'id': 1,
#      'title': 'The Ones Who Walk Away From Omelas',
#      'author': 'Ursula K. Le Guin',
#      'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
#      'published': '1973'},
#     {'id': 2,
#      'title': 'Dhalgren',
#      'author': 'Samuel R. Delany',
#      'first_sentence': 'to wound the autumnal city.',
#      'published': '1975'}
# ]

# Create some test data for our catalog in the form of a list of dictionaries.
# add promptId
responses = [
    {'testId': '1',
     'content': 'Hola',
     'form': 'A',
     'level': 'I',
     'promptId': '11'},
    {'testId': '2',
     'content': 'Hola 2',
     'form': 'A',
     'level': 'N',
     'promptId': '12'},
    {'testId': '3',
     'content': 'Hola 3',
     'form': 'B',
     'level': 'I',
     'promptId': '13'},
    {'testId': '4',
     'content': 'Hola 4',
     'form': 'B',
     'level': 'A',
     'promptId': '14'}
]

# strResponse = json.dumps(responses)
# print(strResponse)
# strEncode=b64encode(strResponse.encode('utf-8'))
# print(strEncode)
# strDecode=b64decode(strEncode).decode('utf-8')
# print(strDecode)
#
# responsesObject=json.loads(strDecode)
# print(str(responsesObject))


@app.route('/', methods=['GET'])
def home():
    return '''<h1>LTI</h1>
<p>LTI request.</p>'''
#
#
# @app.route('/api/v1/resources/books/all', methods=['GET'])
# def api_all():
#     return jsonify(books)
#
#
@app.route('/responses', methods=['GET'])
def api_jsonData():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    try:
        if 'jsonData' in request.args:
            strJson = str(request.args['jsonData'])
            # strDecode = b64decode(strJson).decode('utf-8')
            responseArr = json.loads(strJson)


        else:
            return "Error: No id field provided. Please specify an id."

        # Create an empty list for our results
        results = []

        # Loop through the data and match results that fit the requested ID.
        # IDs are unique, but other fields might return many results
        for responseItem in responseArr:
            responseItem['score']='MF'
            results.append(responseItem)
    except Exception as e:
        string_error = traceback.format_exc()
        print(string_error)
        return string_error


    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)
#
app.run(port=5000)