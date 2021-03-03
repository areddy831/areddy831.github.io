import os
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from flask import Flask, redirect, url_for, request, render_template

# define a Flask app
app = Flask(__name__)
MODEL_VGG16 = load_model('models/Model.h5')
graph = tf.get_default_graph()

print('Successfully loaded VGG19 model...')
print('Visit http://127.0.0.1:5000')

def model_predict(img_path):
    '''
        helper method to process an uploaded image
    '''
    image = load_img(img_path, target_size=(300, 300))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    global graph
    with graph.as_default():
        preds = MODEL_VGG16.predict(image)

    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']

        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        # make prediction about this image's class
        preds = model_predict(file_path)

        pred_class = decode_predictions(preds, top=10)
        result = str(pred_class[0][0][1])
        print('[PREDICTED CLASSES]: {}'.format(pred_class))
        print('[RESULT]: {}'.format(result))

        return result

    return None


if __name__ == '__main__':
    app.run(port=5000, debug=True)
