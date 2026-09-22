import {IT_MODEL, loadGemmaDecision} from './transformers-adapter.mjs';
import {classify, parseCandidates} from './classifier.mjs';

const $ = id => document.getElementById(id);
let loaded;
let latest;
$('modelId').value = IT_MODEL;

function status(message) { $('status').textContent = message; }

async function getModel(id) {
  if (loaded?.id === id) return loaded;
  if (loaded) { await loaded.model.dispose(); loaded = undefined; }
  status('モデルを読み込み中… 数 GB の転送が必要です。');
  const bundle = await loadGemmaDecision(id, info => {
    if (info.status === 'progress_total') status(`モデルを読み込み中… ${Math.round(info.progress)}%`);
  });
  loaded = {id, ...bundle};
  return loaded;
}

$('form').addEventListener('submit', async event => {
  event.preventDefault();
  $('run').disabled = true;
  try {
    if (!navigator.gpu) throw new Error('WebGPU 対応ブラウザが必要です');
    const candidates = parseCandidates($('candidates').value);
    const modelId = $('modelId').value.trim();
    if (!modelId) throw new Error('モデル ID を入力してください');
    const {decision, processor} = await getModel(modelId);
    status('判定中…');
    latest = await classify({decision, processor, candidates,
      input: $('input').value, question: $('question').value, examples: $('examples').value});
    render(latest);
    status('完了。モデルは次の判定にも再利用します。');
  } catch (error) {
    console.error(error);
    status(`エラー: ${error.message ?? String(error)}`);
  } finally { $('run').disabled = false; }
});

function render(result) {
  $('result').hidden = false;
  $('prediction').textContent = result.prediction;
  $('scores').replaceChildren();
  for (const item of result.scores) {
    const row = document.createElement('div');
    row.className = `score${item.name === result.prediction ? ' winner' : ''}`;
    const head = document.createElement('div');
    head.className = 'score-head';
    const name = document.createElement('span');
    name.textContent = `${item.label}. ${item.name}`;
    const value = document.createElement('code');
    value.textContent = `${(item.restrictedProbability * 100).toFixed(1)}% · logit ${item.logit.toFixed(2)}`;
    head.append(name, value);
    const track = document.createElement('div');
    track.className = 'track';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = `${Math.max(0, item.restrictedProbability * 100)}%`;
    track.append(fill);
    row.append(head, track);
    $('scores').append(row);
  }
  $('timing').textContent = `モデル実行: ${(result.timing_ms.forward / 1000).toFixed(2)} 秒`;
}

$('download').addEventListener('click', () => {
  if (!latest) return;
  const blob = new Blob([JSON.stringify(latest, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'jev-classification.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
