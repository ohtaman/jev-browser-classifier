import {AutoProcessor, Gemma4ForCausalLM, Tensor, env} from '@huggingface/transformers';
import {TransformersDecision} from './transformers-decision.mjs';

export const IT_MODEL = 'onnx-community/gemma-4-E2B-it-ONNX';

/** modelId must contain a Transformers.js-compatible Gemma 4 ONNX export. */
export async function loadGemmaDecision(modelId, onProgress = () => {}) {
  const local = modelId.startsWith('local:');
  const id = local ? modelId.slice('local:'.length) : modelId;
  const previous = {path: env.localModelPath, local: env.allowLocalModels,
    remote: env.allowRemoteModels,
    browserCache: env.useBrowserCache};
  // These multi-gigabyte models are served from diamond and should not fill
  // the workstation's limited browser cache during the comparison.
  env.useBrowserCache = false;
  if (local) {
    env.localModelPath = '/models/';
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
  }
  let processor;
  let model;
  try {
    processor = await AutoProcessor.from_pretrained(id);
    // Loading as CausalLM selects text-only sessions (decoder + embedding).
    model = await Gemma4ForCausalLM.from_pretrained(id, {
      dtype: 'q4f16', device: 'webgpu', progress_callback: onProgress,
    });
  } finally {
    env.localModelPath = previous.path;
    env.allowLocalModels = previous.local;
    env.allowRemoteModels = previous.remote;
    env.useBrowserCache = previous.browserCache;
  }
  const tokenize = text => processor(text, null, null, {add_special_tokens: false});
  const decision = new TransformersDecision({
    modelId, tokenize,
    async forward(input) {
      const output = await model({...input,
        num_logits_to_keep: new Tensor('int64', [1n], [])});
      return output.logits;
    },
  });
  return {decision, processor, model};
}

export function formatCase(item, variant, processor) {
  const body = `${item.examples ?? ''}\n\n## questions\n\n${item.question}`;
  if (variant === 'it') {
    const chat = processor.apply_chat_template(
      [{role: 'user', content: [{type: 'text', text: body}]}],
      {enable_thinking: false, add_generation_prompt: true});
    return `${chat}<output>\n`;
  }
  return `${processor.tokenizer.bos_token ?? '<bos>'}${body}\n<output>\n`;
}
