# nodered-tfjs
Node-RED modules for TensorFlow JavaScript models and data processing.

Instruction:
1. ```git clone https://github.com/tedhtchang/bert-sentiment-tfjs.git```

2. ```cd bert-sentiment-tfjs; npm install; cp patch/tf-core.* node_modules/@tensorflow/tfjs-core/dist/; npm run build; cd ..```

3. ```cd bert-sentiment; npm install; cd ..```

4. ```cd bert-tokenizer; npm install; cd ..```

5. Make sure http://localhost:3000/model/model.json and http://localhost:3000/vocab.json are available. Move the tfjs web friendly model files under bert-sentiment-tfjs/public/model/. In another terminal start the server ```cd bert-sentiment-tfjs; npm run start```.

6. ```npm install; npm run start```
