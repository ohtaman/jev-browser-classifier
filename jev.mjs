import {IT_MODEL, loadGemmaDecision} from './transformers-adapter.mjs';
import {classify, normalizeCandidates} from './classifier.mjs';

/**
 * Load a Transformers.js-compatible Gemma 4 ONNX model for choice scoring.
 * Call dispose() when the classifier is no longer needed.
 */
export async function loadJev({modelId = IT_MODEL, onProgress} = {}) {
  if (typeof modelId !== 'string' || !modelId.trim()) {
    throw new Error('modelId を指定してください');
  }
  const {decision, processor, model} = await loadGemmaDecision(modelId, onProgress);
  return createJevClassifier({decision, processor, model});
}

/** Wrap an already loaded Gemma scorer. Also useful for custom loaders. */
export function createJevClassifier({decision, processor, model}) {
  if (!decision?.choice || !processor?.apply_chat_template || !model?.dispose) {
    throw new Error('decision、processor、model が必要です');
  }
  let disposed = false;
  return {
    get modelId() { return decision.modelId; },
    async classify({text, candidates, question = '最も適切なカテゴリは？', examples = ''} = {}) {
      if (disposed) throw new Error('分類器は dispose 済みです');
      if (typeof text !== 'string' || typeof question !== 'string' ||
          typeof examples !== 'string') {
        throw new Error('text、question、examples は文字列で指定してください');
      }
      return classify({decision, processor, input: text, question,
        candidates: normalizeCandidates(candidates), examples});
    },
    async dispose() {
      if (disposed) return;
      disposed = true;
      await model.dispose();
    },
  };
}
