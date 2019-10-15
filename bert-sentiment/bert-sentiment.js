module.exports = function(RED) {
    function BertSentimentNode(config) {
        RED.nodes.createNode(this,config);
        var node = this;
        node.modelfile = config.model;
        var tf = require('@tensorflow/tfjs');
        
        console.log("TANGO:  loading BERT ...");
        console.log(node.modelfile);
        tf.loadGraphModel(node.modelfile).then( function (model) {
        	node.loaded_model = model;
        	console.log("TANGO:  model loaded ...");
        });
        
        
        node.on('input', function(msg) {
        	console.log("TANGO:  try inference ...");
            // console.log(node.loaded_model);
            
        	// Novel input lengths force recompilation which slows down inference, 
        	// so it's a good idea to enforce a max length.
        	const MAX_INPUT_LENGTH = 15; 
        	
        	// This is the tokenization of '[CLS] Hello, my dog is cute. [SEP]'.
        	//const input_ids = tf.tensor1d(
        	//    [101, 7592, 1010, 2026, 3899, 2003, 10140, 1012, 102, 0, 0, 0, 0, 0, 0], 'int32').pad([[0,128-MAX_INPUT_LENGTH]]).expandDims();

        	//const segment_ids = tf.tensor1d(
        	//    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'int32').pad([[0, 128 - MAX_INPUT_LENGTH]]).expandDims();

        	//const input_mask = tf.tensor1d(
        	//    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 'int32').pad([[0, 128 - MAX_INPUT_LENGTH]]).expandDims(0);

        	console.log("Received tokens:  ");
        	console.log(msg.payload);
        	
        	// Pad the tokens received up to the MAX_INPUT_LENGTH
        	rcv_tokens = JSON.parse("[" + msg.payload + "]");
        	padded_tokens = rcv_tokens.slice();
        	if (padded_tokens.length < MAX_INPUT_LENGTH) {
        		padded_tokens.length = MAX_INPUT_LENGTH;
        		padded_tokens.fill(0,rcv_tokens.length,MAX_INPUT_LENGTH);
        	}
        	console.log(padded_tokens);
        	const input_ids = tf.tensor1d(
        			padded_tokens, 'int32').pad([[0,128-MAX_INPUT_LENGTH]]).expandDims();
        	
        	// Create the segment ID
        	segment = new Array(MAX_INPUT_LENGTH).fill(0);
        	console.log(segment);
        	const segment_ids = tf.tensor1d(
        	    segment, 'int32').pad([[0, 128 - MAX_INPUT_LENGTH]]).expandDims();

        	// Create the input mask:  1 for the tokens received, padded with 0 up to MAX_INPUT_LENGTH
        	mask = segment.fill(1,0,rcv_tokens.length);
        	const input_mask = tf.tensor1d(
        	    mask, 'int32').pad([[0, 128 - MAX_INPUT_LENGTH]]).expandDims(0);
        	console.log(mask);
        	
        	
        	//const layer = node.loaded_model.execute({'segment_ids_1': segment_ids, 'input_ids_1': input_ids, 'input_mask_1': input_mask},
            //	'loss/Softmax');
        	const layer = node.loaded_model.predict({'segment_ids_1': segment_ids, 'input_ids_1': input_ids, 'input_mask_1': input_mask});
        	
        	// console.log(JSON.stringify(layer.arraySync()[0]));
        	//console.log(layer);
        	
        	// msg.payload = "Hello Tango";
        	msg.payload = JSON.stringify(layer.arraySync()[0]);
            node.send(msg);
        });
    }
    RED.nodes.registerType("bert-sentiment",BertSentimentNode);
}
