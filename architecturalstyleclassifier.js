// tensorflow.js 1.0.0
const MODEL_URL = '/web_model/model.json'

// https://js.tensorflow.org/api/1.0.0/#loadGraphModel
const model = await tf.loadGraphModel(MODEL_URL)

// create a tensor from an image - tensorflow.js 1.0.0
// https://js.tensorflow.org/api/1.0.0/#browser.fromPixels
const imageTensor = tf.browser.fromPixels(imageElement)

// insert a dimension into the tensor's shape
const preprocessedInput = imageTensor.expandDims(300,300,3)


// generates output prediction
const prediction = model.predict(preprocessedInput)
