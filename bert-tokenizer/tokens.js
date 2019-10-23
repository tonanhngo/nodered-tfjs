module.exports = function(RED) {
    function TokensNode(config) {
        RED.nodes.createNode(this,config);
				var tf = require('@tensorflow/tfjs');
				var tokenizer = require('bert-sentiment-tfjs');
				var node = this;
				tokens = new tokenizer.default(true)
				tokens.init('http://localhost:3000/vocab.json');

        node.on('input', function(msg) {
					console.log("Payload: " + msg.payload);
					tokens.inputFeature(msg.payload).then((result) => {
						msg.inputFeature = result;
						node.send(msg);
					}).catch((err) => {
						console.log("Something went wrong")
					});
        });
    }
    RED.nodes.registerType("tokens",TokensNode);
}
