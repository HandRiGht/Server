from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import json
import base64
import time
from Segmentation import Words


app = Flask(__name__)
port = 1337
word = Words("C:\\Users\\sohai\\PycharmProjects\\Server\\words_rec.jpg")


@app.route('/mobile/image', methods=['POST'])
def image():
    file = open("words_rec.jpg", "wb")
    #y = request.get_data()
    #print(y)

    temp = request.get_data().decode("utf-8").split(":")[1]
    temp = temp.replace("\"", "")
    temp = temp.replace("}", "")

    temp2 = request.get_data().decode("utf-8").split(":")[0]
    temp2 = temp2.replace("\"", "")
    temp2 = temp2.replace("}", "")
    temp2 = temp2.replace("{", "")
    word.set_cache(temp2)

    file.write(base64.decodebytes(temp.encode("utf-8")))
    file.close()
    return word.detect_document()


@app.route('/mobile/processed')
def run():
    return word.detect_document()


@app.route('/mobile/score')
def play():
    return str(word.max_score)



@app.route('/mobile/check')
def check():
    word.calculateIncorrectLetters(word.wordCache)
    #with open("Updated_change.jpg", "rb") as image_file:
    #    encoded_string = base64.b64encode(image_file.read())
    #eturn encoded_string
    return send_from_directory("", "Updated_change.jpg")


@app.route('/handright')
def handright():
    return {"Name": "John Doe", "Description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."}


@app.route('/handright/feed')
def feed():
    return send_from_directory("", "words_rec.jpg")



if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = port)