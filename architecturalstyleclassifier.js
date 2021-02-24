/* global tf, Image, FileReader, ImageData, fetch */

const modelUrl = '/web_model/model.json'

const imageSize = 300

let targetSize = { w: imageSize, h: imageSize }
let model
let imageElement
let colorMap

/**
 * load the TensorFlow.js model
 */
window.loadModel = async function () {
  disableElements()
  message('loading model...')

  let start = (new Date()).getTime()

  // https://js.tensorflow.org/api/1.1.2/#loadGraphModel
  model = await tf.loadGraphModel(modelUrl)

  let end = (new Date()).getTime()

  message(model.modelUrl)
  message(`model loaded in ${(end - start) / 1000} secs`, true)
  enableElements()
}

/**
 * handle image upload
 *
 * @param {DOM Node} input - the image file upload element
 */
window.loadImage = function (input) {
  if (input.files && input.files[0]) {
    disableElements()
    message('resizing image...')

    let reader = new FileReader()

    reader.onload = function (e) {
      let src = e.target.result

      document.getElementById('canvasimage').getContext('2d').clearRect(0, 0, targetSize.w, targetSize.h)
      document.getElementById('canvassegments').getContext('2d').clearRect(0, 0, targetSize.w, targetSize.h)

      imageElement = new Image()
      imageElement.src = src

      imageElement.onload = function () {
        let resizeRatio = imageSize / Math.max(imageElement.width, imageElement.height)
        targetSize.w = Math.round(resizeRatio * imageElement.width)
        targetSize.h = Math.round(resizeRatio * imageElement.height)

        let origSize = {
          w: imageElement.width,
          h: imageElement.height
        }
        imageElement.width = targetSize.w
        imageElement.height = targetSize.h

        let canvas = document.getElementById('canvasimage')
        canvas.width = targetSize.w
        canvas.height = targetSize.w
        canvas
          .getContext('2d')
          .drawImage(imageElement, 0, 0, targetSize.w, targetSize.h)

        message(`resized from ${origSize.w} x ${origSize.h} to ${targetSize.w} x ${targetSize.h}`)
        enableElements()
      }
    }

    reader.readAsDataURL(input.files[0])
  } else {
    message('no image uploaded', true)
  }
}

/**
 * run the model and get a prediction
 */
window.runModel = async function () {
  if (imageElement) {
    disableElements()
    message('running inference...')

    let img = preprocessInput(imageElement)
    console.log('model.predict (input):', img.dataSync())

    let start = (new Date()).getTime()

    // https://js.tensorflow.org/api/latest/#tf.Model.predict
    const output = model.predict(img)

    let end = (new Date()).getTime()

    console.log('model.predict (output):', output.dataSync())
    await processOutput(output)

    message(`inference ran in ${(end - start) / 1000} secs`, true)
    enableElements()
  } else {
    message('no image available', true)
  }
}

/**
 * convert image to Tensor input required by the model
 *
 * @param {HTMLImageElement} imageInput - the image element
 */
function preprocessInput (imageInput) {
  console.log('preprocessInput started')

  let inputTensor = tf.browser.fromPixels(imageInput)

  // https://js.tensorflow.org/api/latest/#expandDims
  let preprocessed = inputTensor.expandDims()

  console.log('preprocessInput completed:', preprocessed)
  return preprocessed
}
