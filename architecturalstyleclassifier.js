async function run(){
  const image = tf.browser.fromPixels(imgcanvas);
  const resized_image =
       tf.image.resizeBilinear(image, [300,300]).toFloat();
  const offset = tf.scalar(255.0);
  const normalized = tf.scalar(1.0).sub(resized_image.div(offset));
  const MODEL_URL = 'areddy831.github.io/web_model/model.json';
  const model = await tf.loadLayersModel(MODEL_URL);
  const result = model.predict(batchedImage);
  result.print();
}
run();
