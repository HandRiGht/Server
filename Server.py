from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

app = Flask(__name__)
port = 1337


@app.route('/mobile/image', methods=['POST'])
def image():
    with open("words.jpg", "wb") as outputFile:
        outputFile.write(request.data)
        return "Image received"


@app.route('/handright')
def handright():
    return {"Name": "John Doe", "Description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."}


@app.route('/handright/feed')
def feed():
    return send_from_directory("", "words.png")



if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = port)