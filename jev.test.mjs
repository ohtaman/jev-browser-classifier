import assert from 'node:assert/strict';
import test from 'node:test';
import {createJevClassifier} from 'jev-browser-classifier';

test('classifies named choices with one model call and disposes once', async () => {
  let calls = 0;
  let disposals = 0;
  const jev = createJevClassifier({
    decision: {modelId: 'fake', async choice(prompt, labels) {
      calls++;
      assert.match(prompt, /<output>\n$/);
      assert.deepEqual(labels, ['1', '2', '3']);
      return {modelId: 'fake', winner: '2', generatedTokenCount: 0,
        options: labels.map((label, index) => ({label, logit: index,
          restrictedProbability: index / 3})), timingMs: {forward: 20}};
    }},
    processor: {apply_chat_template: () => '<chat>'},
    model: {async dispose() { disposals++; }},
  });
  const result = await jev.classify({text: '二重請求', candidates: [
    '配送', {name: '請求', description: '料金と返金'}, '技術',
  ]});
  assert.equal(result.prediction, '請求');
  assert.equal(result.scores[1].description, '料金と返金');
  assert.equal(result.generated_tokens, 0);
  assert.equal(calls, 1);
  await jev.dispose();
  await jev.dispose();
  assert.equal(disposals, 1);
  await assert.rejects(() => jev.classify({text: 'x', candidates: ['a', 'b']}), /dispose/);
});

test('rejects invalid input before calling the model', async () => {
  let called = false;
  const jev = createJevClassifier({decision: {choice: async () => { called = true; }},
    processor: {apply_chat_template: () => ''}, model: {dispose: async () => {}}});
  await assert.rejects(() => jev.classify({text: '', candidates: ['a', 'b']}), /入力/);
  await assert.rejects(() => jev.classify({text: 'x', candidates: ['a']}), /2〜64/);
  assert.equal(called, false);
});
