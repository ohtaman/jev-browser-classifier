import assert from 'node:assert/strict';
import test from 'node:test';
import {buildClassifierBody, classify, parseCandidates} from './classifier.mjs';

test('maps arbitrary categories to unique fixed labels', () => {
  const choices = parseCandidates('配送: 荷物について\n請求: 支払いについて\n技術');
  assert.deepEqual(choices.map(item => item.label), ['1', '2', '3']);
  assert.match(buildClassifierBody({input: '返金したい', question: '最も近い分類は？', candidates: choices}),
    /2\. 請求 — 支払いについて/);
});

test('returns named classification and every supplied score', async () => {
  const candidates = parseCandidates('配送\n請求');
  let received;
  const result = await classify({input: '請求ミス', question: '分類は？', candidates,
    processor: {apply_chat_template: () => '<chat>'},
    decision: {choice: async (prompt, labels) => {
      received = {prompt, labels};
      return {modelId: 'test', winner: '2', options: [
        {label: '1', logit: 1, restrictedProbability: 0.1},
        {label: '2', logit: 3, restrictedProbability: 0.9},
      ], generatedTokenCount: 0, timingMs: {total: 10}};
    }}});
  assert.deepEqual(received.labels, ['1', '2']);
  assert.match(received.prompt, /<output>\n$/);
  assert.equal(result.prediction, '請求');
  assert.equal(result.scores.length, 2);
});

test('rejects ambiguous or unsupported category lists', () => {
  assert.throws(() => parseCandidates('one'), /2〜64/);
  assert.throws(() => parseCandidates('one\none'), /重複/);
});
