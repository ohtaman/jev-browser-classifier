# Jev browser classifier

## JavaScript API

Install the source package in a browser app with a bundler such as Vite:

```sh
npm install github:ohtaman/jev-browser-classifier
```

```js
import {loadJev} from 'jev-browser-classifier';

const jev = await loadJev(); // Jev Gemma 4 E2B IT choice-64 ONNX, WebGPU
const result = await jev.classify({
  text: '注文した荷物がまだ届いていません。',
  candidates: ['配送', '請求', '技術'],
});
console.log(result.prediction, result.scores);
await jev.dispose();
```

`candidates` accepts 2–64 strings or objects such as `{name: '配送', description: '荷物の未着'}`. Optional `question` and `examples` customize the prompt. `loadJev({modelId, onProgress})` can select a compatible Gemma 4 ONNX model and report download progress. Load once, classify multiple texts, then dispose of the model. The returned `scores` include each candidate's raw logit and a value normalized across the supplied candidates. The normalized values are not calibrated probabilities.

The default model is [ohtaman/jev-gemma-4-E2B-it-choice-64](https://huggingface.co/ohtaman/jev-gemma-4-E2B-it-choice-64), a text-only 64-output ONNX artifact derived from Gemma 4 E2B IT. `loadJev({modelId})` can also use a full-vocabulary compatible Gemma 4 ONNX model; it reads the choice manifest when present. This API runs in a WebGPU browser. Model loading transfers several GB from Hugging Face and may happen again after a page reload.

A browser based general purpose classifier using Gemma 4 E2B IT and Transformers.js. Enter text, a question, and 2–64 named categories. The app compares the first answer token's logits in one forward pass. It does not generate an answer or call a classification API.

**Open the site:** https://ohtaman.github.io/jev-browser-classifier/

Requires a WebGPU browser. Loading the default model transfers several GB from Hugging Face. The model runs in the browser and is kept in memory for further classifications on the same page. This site does not host model weights.

## Run locally

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/`.

## How it works

The default model is [onnx-community/gemma-4-E2B-it-ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX), a Transformers.js ONNX export of [Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-E2B-it). Candidate names and descriptions are written into the prompt. They are assigned fixed labels `1–9`, `0`, `A–Z`, `a–z`, `@`, `#`. Before inference, the app checks that each label appends exactly one unique token at the answer boundary. It reads their logits from a single forward pass and applies softmax only over those choices. These values are relative choice scores, **not calibrated probabilities**.

The model remains a general language model. This app does not fine tune weights or train a dedicated classifier. Classification quality depends on task, prompt, categories, and their order. The browser experiment has been checked on WebGPU with the same ONNX model files served locally; the published URL uses the public Hugging Face files.

## Development

```sh
npm test
npm run build
```
