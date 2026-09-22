/** Compare the first answer token's logits. No generation or sampling occurs. */
export class TransformersDecision {
  constructor({tokenize, forward, modelId, tokenToLogitIndex = null}) {
    this.tokenize = tokenize;
    this.forward = forward;
    this.modelId = modelId;
    this.tokenToLogitIndex = tokenToLogitIndex;
  }

  async choice(prompt, labels = ['1', '2', '3']) {
    if (!prompt || !labels.length) throw new Error('Prompt and labels are required');
    const start = performance.now();
    const input = await this.tokenize(prompt);
    const ids = Array.from(input.input_ids.data, Number);
    if (!ids.length) throw new Error('Empty prompt tokenization');
    const labelIds = [];
    for (const label of labels) {
      const withLabel = await this.tokenize(prompt + label);
      const next = Array.from(withLabel.input_ids.data, Number);
      if (next.length !== ids.length + 1 ||
          ids.some((id, index) => id !== next[index])) {
        throw new Error(`unsupported_token_boundary: ${JSON.stringify(label)}`);
      }
      labelIds.push(next.at(-1));
    }
    if (new Set(labelIds).size !== labelIds.length) {
      throw new Error('Labels resolve to duplicate token IDs');
    }
    const indices = labelIds.map(id => this.tokenToLogitIndex ?
      this.tokenToLogitIndex.get(id) : id);
    if (indices.some(index => !Number.isInteger(index) || index < 0)) {
      throw new Error('One or more labels are missing from the model output head');
    }
    const tokenizeMs = performance.now() - start;
    const rawLogits = await this.forward(input);
    const logits = rawLogits.type === 'float16' ? rawLogits.to('float32') : rawLogits;
    const {dims, data} = logits;
    if (dims.length !== 3 || dims[0] !== 1 || dims[1] < 1) {
      throw new Error(`Unexpected logits shape: ${JSON.stringify(dims)}`);
    }
    const offset = (dims[1] - 1) * dims[2];
    if (indices.some(index => index >= dims[2])) {
      throw new Error('Model logits width does not include a requested label');
    }
    const values = indices.map(index => Number(data[offset + index]));
    if (values.some(value => !Number.isFinite(value))) {
      throw new Error('Missing or nonfinite candidate logit');
    }
    const max = Math.max(...values);
    const weights = values.map(value => Math.exp(value - max));
    const total = weights.reduce((a, b) => a + b, 0);
    const options = labels.map((label, index) => ({
      label, tokenId: labelIds[index], logit: values[index],
      restrictedProbability: weights[index] / total,
    }));
    const winner = options.reduce((best, item) =>
      item.logit > best.logit ? item : best);
    return {modelId: this.modelId, prompt, winner: winner.label, options,
      generatedTokenCount: 0,
      timingMs: {tokenize: tokenizeMs, forward: performance.now() - start - tokenizeMs,
        total: performance.now() - start}};
  }
}

export async function compareModels({cases, models}) {
  if (!Array.isArray(cases) || !cases.length) throw new Error('Cases are required');
  const results = [];
  for (const {id, decision, formatPrompt} of models) {
    const rows = [];
    for (const item of cases) {
      const output = await decision.choice(formatPrompt(item),
        item.labels ?? ['1', '2', '3']);
      rows.push({id: item.id, gold: item.gold, correct: output.winner === item.gold,
        ...output});
    }
    results.push({id, accuracy: rows.filter(row => row.correct).length / rows.length,
      meanForwardMs: rows.reduce((sum, row) => sum + row.timingMs.forward, 0) / rows.length,
      rows});
  }
  return results;
}
