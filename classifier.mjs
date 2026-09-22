// Stable single-token labels for Gemma 4 at the answer boundary. The runtime
// verifies the token boundary again for every prompt before inference.
export const CHOICE_LABELS = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@#';

export function parseCandidates(source) {
  const candidates = source.split(/\r?\n/).map(line => line.trim()).filter(Boolean).map((line, index) => {
    const separator = line.indexOf(':');
    const name = (separator < 0 ? line : line.slice(0, separator)).trim();
    const description = (separator < 0 ? line : line.slice(separator + 1)).trim();
    if (!name) throw new Error(`${index + 1}行目のカテゴリ名が空です`);
    return {name, description};
  });
  if (candidates.length < 2 || candidates.length > CHOICE_LABELS.length) {
    throw new Error(`カテゴリは2〜${CHOICE_LABELS.length}個指定してください`);
  }
  if (new Set(candidates.map(item => item.name)).size !== candidates.length) {
    throw new Error('カテゴリ名が重複しています');
  }
  return candidates.map((candidate, index) => ({...candidate, label: CHOICE_LABELS[index]}));
}

export function buildClassifierBody({input, question, candidates, examples = ''}) {
  if (!input.trim()) throw new Error('判定するテキストを入力してください');
  if (!question.trim()) throw new Error('判定基準を入力してください');
  const choices = candidates.map(({label, name, description}) =>
    `${label}. ${name}${description ? ` — ${description}` : ''}`).join('\n');
  const exampleBlock = examples.trim() ? `## few shot examples\n${examples.trim()}\n\n` : '';
  return `${exampleBlock}## questions\n<input>\nstate: ${input.trim()}\nquestion1: ${question.trim()}\n${choices}\n</input>`;
}

export function formatClassifierPrompt(body, processor) {
  const chat = processor.apply_chat_template(
    [{role: 'user', content: [{type: 'text', text: body}]}],
    {enable_thinking: false, add_generation_prompt: true});
  return `${chat}<output>\n`;
}

export async function classify({decision, processor, input, question, candidates, examples}) {
  const body = buildClassifierBody({input, question, candidates, examples});
  const prompt = formatClassifierPrompt(body, processor);
  const result = await decision.choice(prompt, candidates.map(item => item.label));
  return {
    model: result.modelId,
    input,
    question,
    prediction: candidates.find(item => item.label === result.winner).name,
    scores: candidates.map((item, index) => ({...item, ...result.options[index]})),
    generated_tokens: result.generatedTokenCount,
    timing_ms: result.timingMs,
    note: 'restrictedProbability is normalized across the supplied choices; it is not a calibrated confidence.',
  };
}
