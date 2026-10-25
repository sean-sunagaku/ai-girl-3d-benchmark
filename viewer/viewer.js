const $ = (selector) => document.querySelector(selector);
const model = $('#model');
const render = $('#render');
const loading = $('#loading');
const viewLabels = { '3d': 'INTERACTIVE 3D', beauty: 'ポートレート', front: '正面 · 0°', 'three-quarter': '斜め · 45°', side: '横 · 90°', back: '背面 · 180°' };
let record;
let stage = 'procedural';
let view = '3d';
let modelLoaded = false;
const narrowScreen = matchMedia('(max-width: 620px)');
const initialOrbit = entry => narrowScreen.matches ? (entry.camera_orbit_mobile || '0deg 80deg 105%') : (entry.camera_orbit || '0deg 80deg 80%');

function selectView(next) {
  view = next;
  document.querySelectorAll('[data-view]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
  const is3d = view === '3d';
  model.hidden = !is3d;
  render.hidden = is3d;
  $('#reset-camera').hidden = !is3d;
  $('#view-label').textContent = viewLabels[view];
  $('#interaction-note').textContent = is3d ? 'ドラッグで回転 · ホイール／ピンチで拡大縮小 · 2本指で移動' : '保存された同じBlenderシーンからのレンダー画像';
  loading.hidden = !is3d || modelLoaded;
  if (!is3d) {
    render.src = record.stages[stage].renders[view];
    render.alt = `Kimi K3.1 Pro · ${stage === 'procedural' ? '初回' : '写真テクスチャ許可版'} · ${viewLabels[view]}`;
  }
}

function selectStage(next) {
  if (!record.stages[next]) return;
  stage = next;
  const entry = record.stages[stage];
  document.querySelectorAll('[data-stage]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.stage === stage)));
  $('#stage-number').textContent = stage === 'procedural' ? '01' : '02';
  $('#stage-description').textContent = stage === 'procedural' ? '写真テクスチャを使わず、形の参照から制作。' : 'Kimi自身の初回モデルを引き継ぎ、写真由来のテクスチャも許可した仕上げ。';
  modelLoaded = false;
  loading.textContent = '3Dモデルを読み込んでいます…';
  model.src = entry.glb;
  model.poster = entry.renders.beauty;
  model.cameraOrbit = initialOrbit(entry);
  document.querySelectorAll('[data-view] img').forEach(image => { image.src = entry.renders[image.closest('button').dataset.view]; });
  for (const [selector, key] of Object.entries({'#download-glb':'glb','#download-blend':'blend','#source-code':'code','#run-readme':'readme','#scene-report':'validation'})) {
    if (entry[key]) $(selector).href = entry[key];
    else $(selector).removeAttribute('href');
  }
  $('#run-record').href = record.run_record;
  $('#cost').textContent = record.cost_note;
  selectView(view);
}

model.addEventListener('load', () => { modelLoaded = true; loading.hidden = true; });
model.addEventListener('error', () => {
  loading.textContent = '3Dを読み込めませんでした。下の画像からレンダーを確認できます。';
  loading.hidden = view !== '3d';
});
render.addEventListener('error', () => { loading.textContent = '画像を読み込めませんでした。制作記録からファイルを確認してください。'; loading.hidden = false; });
$('#reset-camera').addEventListener('click', () => { if (!record) return; model.cameraOrbit = initialOrbit(record.stages[stage]); model.fieldOfView = 'auto'; model.jumpCameraToGoal(); });
narrowScreen.addEventListener('change', () => { if (record) model.cameraOrbit = initialOrbit(record.stages[stage]); });
document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => { if (record) selectStage(button.dataset.stage); }));
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => { if (record) selectView(button.dataset.view); }));

async function init() {
  document.querySelectorAll('button').forEach(button => { button.disabled = true; });
  try {
    const response = await fetch('results.json');
    if (!response.ok) throw new Error('Result record unavailable');
    record = await response.json();
    selectStage(stage);
    document.querySelectorAll('button').forEach(button => { button.disabled = false; });
    document.querySelectorAll('[data-stage]').forEach(button => {
      if (!record.stages[button.dataset.stage]) {
        button.disabled = true;
        button.textContent = '写真テクスチャ許可版 · 未実行';
        button.title = record.pending_note;
      }
    });
    $('#pending-stage').textContent = record.pending_note || '';
    // A script failure still leaves all genuine render images available.
    setTimeout(() => { if (!customElements.get('model-viewer')) { loading.textContent = '3Dビューアーを読み込めませんでした。下のレンダー画像を選んでください。'; loading.hidden = view !== '3d'; } }, 15000);
  } catch {
    loading.textContent = '制作記録を取得できません。リポジトリのルートでHTTPサーバーを起動して開いてください。';
  }
}
init();
