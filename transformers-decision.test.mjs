import assert from 'node:assert/strict';
import test from 'node:test';
import {TransformersDecision, compareModels} from './transformers-decision.mjs';

function decision(scores = {1: 2, 2: 5, 3: 1}) {
  let forwards = 0;
  const tokenize = async text => {
    const label = text.match(/[123]$/)?.[0];
    const stem = label ? text.slice(0, -1) : text;
    return {input_ids: {data: BigInt64Array.from([
      ...Array.from(stem).map(char => BigInt(char.codePointAt(0))),
      ...(label ? [BigInt(label)] : []),
    ])}};
  };
  const model = new TransformersDecision({modelId: 'fake', tokenize,
    async forward() {
      forwards++;
      const data = new Float32Array(10);
      for (const [id, score] of Object.entries(scores)) data[Number(id)] = score;
      return {dims: [1, 1, 10], data};
    }});
  return {model, get forwards() { return forwards; }};
}

test('compares first answer token logits with one forward', async () => {
  const fake = decision();
  const result = await fake.model.choice('answer:\n');
  assert.equal(result.winner, '2');
  assert.equal(result.generatedTokenCount, 0);
  assert.equal(fake.forwards, 1);
  assert.deepEqual(result.options.map(option => option.tokenId), [1, 2, 3]);
});

test('rejects a changed token boundary before inference', async () => {
  const fake = decision();
  await assert.rejects(() => fake.model.choice('answer:\n1'),
    /unsupported_token_boundary/);
  assert.equal(fake.forwards, 0);
});

test('reports accuracy for two independent model profiles', async () => {
  const first = decision({1: 5, 2: 1, 3: 0});
  const second = decision({1: 0, 2: 5, 3: 1});
  const results = await compareModels({cases: [{id: 'q', gold: '2'}], models: [
    {id: 'base', decision: first.model, formatPrompt: () => 'Q\n'},
    {id: 'it', decision: second.model, formatPrompt: () => 'Q\n'},
  ]});
  assert.deepEqual(results.map(result => result.accuracy), [0, 1]);
});

test('converts float16 logits before comparing values', async () => {
  const fake = decision();
  fake.model.forward = async () => ({type: 'float16', to(type) {
    assert.equal(type, 'float32');
    const data = new Float32Array(10);
    data[3] = 7;
    return {dims: [1, 1, 10], data};
  }});
  assert.equal((await fake.model.choice('answer:\n')).winner, '3');
});

test('maps token IDs to a compact output head before inference', async () => {
  const fake = decision();
  fake.model.tokenToLogitIndex = new Map([[1, 6], [2, 4], [3, 8]]);
  fake.model.forward = async () => {
    const data = new Float32Array(10);
    data[6] = 2; data[4] = 5; data[8] = 1;
    return {dims: [1, 1, 10], data};
  };
  const result = await fake.model.choice('answer:\n');
  assert.equal(result.winner, '2');
  assert.deepEqual(result.options.map(option => option.logit), [2, 5, 1]);
});
