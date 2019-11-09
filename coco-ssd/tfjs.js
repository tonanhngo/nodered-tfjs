
module.exports = function (RED) {
    function TensorFlowPredict(n) {
        var tf = require('@tensorflow/tfjs-node');
        var fs = require('fs');
        
        RED.nodes.createNode(this, n);
        this.modelfile = n.model;
        //this.modelfile = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
        var node = this;

        async function loadModel(m) {
            await tf.loadLayersModel(m).then(function (model) {
                var shp = model.inputs[0].shape;
                shp.shift();
                node.log("Shape wanted: "+shp);
                shp.unshift(1);
                model.predict(tf.zeros(shp)).dispose();
                node.model = model;
                node.shp = shp;
                node.ready = true;
                node.status({fill:'green', shape:'dot', text:'Model ready'});
            });
        }
        node.status({fill:'yellow', shape:'ring', text:'Loading model...'});
        loadModel(node.modelfile);

        async function imgToTensor(p) {
            // if it's a string assume it's a filename
            if (typeof p === "string") { p = fs.readFileSync(p); }
            const img = tf.node.decodeImage(p,node.shp[3]);
            // rescale the image to fit the wanted shape
            const scaled = tf.image.resizeBilinear(img, [node.shp[1],node.shp[2]], true);
            const offset = tf.scalar(127.5);
            // Normalize the image from [0, 255] to [-1, 1].
            const normalized = scaled.sub(offset).div(offset);
            // extend the tensor to 4d
            return normalized.reshape(node.shp);
        }

        node.on('input', function (msg) {
            msg.inputShape = node.shp;
            try {
                imgToTensor(msg.payload)
                    .then(
                        function(t) {
                            var tensorResult = node.model.predict(t);
                            msg.maxIndex = tensorResult.argMax(1).dataSync()[0];
                            msg.payload = Array.from(tensorResult.dataSync());
                            node.send(msg);
                        },
                        function(e) {
                            node.error("OOPS: "+e,msg);
                        }
                    )
            } catch (error) {
                node.error(error, msg);
            }
        });

        node.on("close", function () {
            node.status({});
        });
    }
    RED.nodes.registerType("tensorflowPredict", TensorFlowPredict);

    function TensorFlowCoCo(n) {
        var fs = require('fs');
        var tf = require('@tensorflow/tfjs-node');
        var cocoSsd = require('@tensorflow-models/coco-ssd');
        
        RED.nodes.createNode(this, n);
        var node = this;

        async function loadModel() {
            node.model = await cocoSsd.load();
            node.ready = true;
            node.status({fill:'green', shape:'dot', text:'Model ready'});
        }
        node.status({fill:'yellow', shape:'ring', text:'Loading model...'});
        loadModel();

        node.on('input', function (msg) {
            async function reco(img) {
                msg.payload = await node.model.detect(img);
                msg.shape = img.shape;
                msg.classes = {};
                for (var i=0; i<msg.payload.length; i++) {
                    msg.classes[msg.payload[i].class] = (msg.classes[msg.payload[i].class] || 0 ) + 1;
                }
                node.send(msg);
            }
            try {
                if (node.ready) {
                    var p = msg.payload;
                    if (typeof p === "string") { p = fs.readFileSync(p); }
                    reco(tf.node.decodeImage(p));
                }
            } catch (error) {
                node.error(error, msg);
            }
        });

        node.on("close", function () {
            node.status({});
        });
    }
    RED.nodes.registerType("tensorflowCoco", TensorFlowCoCo);
};
