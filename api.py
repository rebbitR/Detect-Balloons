
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import detect_balloon_main

app = Flask(__name__)
CORS(app)

@app.route('/post', methods=['POST'])
def program():
    image_name="image.jpg"
    result_image_name='result_balloon.png'

    file = request.files['myFile']
    file.save(image_name)

    frame_result=detect_balloon_main(image_name)

    return 'http://127.0.0.1:8887/'+'result_balloon.png'

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)










