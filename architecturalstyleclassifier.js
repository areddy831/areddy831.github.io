// load model in js script
(async () => {
  ...
  const model = await tf.loadFrozenModel('saved_model.pb', '/web_model/model.json')
})()
