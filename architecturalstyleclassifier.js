model = await tf.loadModel('web_model/model.json')

function preprocess(img)
{

    //convert the image data to a tensor
    let tensor = tf.fromPixels(img)
    //resize to 300 X 300
    const resized = tf.image.resizeBilinear(tensor, [300, 300]).toFloat()
    // Normalize the image
    const offset = tf.scalar(255.0);
    const normalized = tf.scalar(1.0).sub(resized.div(offset));
    //We add a dimension to get a batch shape
    const batched = normalized.expandDims(0)
    return batched

}


const pred = model.predict(preprocess(img)).dataSync()
