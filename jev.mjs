import {loadGemmaDecision} from './transformers-adapter.mjs';
import {classify, normalizeCandidates} from './classifier.mjs';

export const JEV_MODEL = 'ohtaman/jev-gemma-4-E2B-it-choice-64';

async function loadChoiceManifest(modelId) {
  const local = modelId.startsWith('local:');
  const id = local ? modelId.slice('local:'.length) : modelId;
  const url = local ? `/models/${id}/choice_head_64.json` :
    `https://huggingface.co/${id}/resolve/main/choice_head_64.json`;
  const response = await fetch(url);
  if (response.status === 404) return null; // full-vocabulary ONNX artifact
  if (!response.ok) throw new Error(`choice manifest: HTTP ${response.status}`);
  const manifest = await response.json();
  const {labels, token_ids: ids} = manifest;
  if (!Array.isArray(labels) || !Array.isArray(ids) || labels.length !== 64 ||
      ids.length !== 64 || new Set(ids).size !== 64 ||
      ids.some(id => !Number.isInteger(id) || id < 0)) {
    throw new Error('Invalid 64-label model manifest');
  }
  return new Map(ids.map((id, index) => [id, index]));
}

/**
 * Load a Transformers.js-compatible Gemma 4 ONNX model for choice scoring.
 * Call dispose() when the classifier is no longer needed.
 */
export async function loadJev({modelId = JEV_MODEL, onProgress} = {}) {
  if (typeof modelId !== 'string' || !modelId.trim()) {
    throw new Error('modelId を指定してください');
  }
  const tokenToLogitIndex = await loadChoiceManifest(modelId);
  const {decision, processor, model} = await loadGemmaDecision(modelId, onProgress,
    {tokenToLogitIndex});
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
