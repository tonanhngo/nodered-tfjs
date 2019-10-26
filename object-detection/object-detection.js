module.exports = function(RED) {

  const cocoSsd = require('@tensorflow-models/coco-ssd');
  const tf = require('@tensorflow/tfjs-node');

  function ObjectDetectionNode(config) {
    RED.nodes.createNode(this, config);

    this.modelUrl = config.modelUrl;
    const node = this;

    node.status({fill:'yellow', shape:'dot', text:'Loading model...'});

    cocoSsd.load({modelUrl: node.modelUrl}).then(model => {
      node.loadedModel = model;
      node.status({fill:'green', shape:'dot', text:'Model is ready'});
      console.log('Object Detection Model Loaded.');
    });

    node.on('input', function(msg) {
      const imgTensor = tf.node.decodeImage(new Uint8Array(msg.payload), channels = 3);
      node.loadedModel.detect(imgTensor).then(predictions => {
        msg.payload = predictions;
        node.send(msg);
      });
    });
  }
  RED.nodes.registerType("object-detection", ObjectDetectionNode);
}
