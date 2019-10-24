module.exports = function(RED) {
    function BertSentimentNode(config) {
        RED.nodes.createNode(this,config);
        var node = this;
        node.modelfile = config.model;

				var tf = require('@tensorflow/tfjs-node');
				console.log("Loading model...")
				tf.loadGraphModel('http://localhost:3000/model/model.json').then((model)=>{
					node.loaded_model = model;
					node.status({fill:'green' ,shape:'dot', text:'model is ready'});
					console.log("Model loaded...")
				});
				node.status({fill:'green' ,shape:'dot', text:'model loading'})
        node.on('input', (msg) => {
					res = tf.tidy(() => {
						let pred = node.loaded_model.execute({...msg.inputFeature}, "loss/Softmax");
						return pred.squeeze([0]);
					});
					resarr = res.arraySync()
					msg.payload = {"positive":resarr[0], "negative":resarr[1]};
          node.send(msg);
        });
    }
    RED.nodes.registerType("bert-sentiment",BertSentimentNode);
}
