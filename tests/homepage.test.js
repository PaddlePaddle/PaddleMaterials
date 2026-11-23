const assert = require('node:assert/strict');
const { existsSync, readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const homepageDir = path.join(__dirname, '..', 'docs');
const html = readFileSync(path.join(homepageDir, 'index.html'), 'utf8');
const readme = readFileSync(path.join(__dirname, '..', 'README.md'), 'utf8');
const script = readFileSync(path.join(homepageDir, 'script.js'), 'utf8');

const trainCommand = 'python property_prediction/train.py -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml';
const predictFragments = [
  'python property_prediction/predict.py',
  "--model_name='megnet_mp2018_train_60k_e_form'",
  "--weights_name='best.pdparams'",
  "--cif_file_path='./property_prediction/example_data/cifs/'",
  "--save_path='result.csv'",
];

test('platform signals use the approved Chinese copy', () => {
  assert.match(html, /<div class="eyebrow"><span class="status-dot"><\/span> Open-source AI infrastructure for materials science\.<\/div>/);
  assert.doesNotMatch(html, /OPEN-SOURCE MATERIALS AI/);
  assert.match(html, /<small>预置模型权重和数据集<\/small>/);
  assert.doesNotMatch(html, /Config-driven experiments/);
  assert.match(html, /连接材料结构、性质、电子结构、谱图与模拟/);
  assert.doesNotMatch(html, /连接材料结构、性质、电子态、光谱与模拟/);
  assert.match(html, /<div class="hero-note"><span class="spark">✦<\/span> 从结构到性质，从预测到表征，把研究流程交给模型。<\/div>/);
  assert.doesNotMatch(html, /从结构到性质，从实验到预测，把研究流程交给模型。/);
});

test('workflow uses the approved six-step Chinese copy', () => {
  const expectedSteps = [
    ['配置任务与输入', 'YAML 配置 · 任务脚本 · 结构/谱图/图像输入'],
    ['装配数据与模型', '数据集构建 · 模型构建 · 超参数配置'],
    ['训练或微调模型', '训练与微调 · 检查点管理 · 指标评估'],
    ['加载权重并执行推理', '权重加载 · 输入转换 · 任务级推理'],
    ['接入材料计算链路', '文件读写 · ASE 计算器 · 动力学/结构优化'],
    ['输出结果并迭代实验', '结果导出 · 微调续训 · 迭代实验'],
  ];
  for (const [title, detail] of expectedSteps) {
    assert.ok(html.includes(`<strong>${title}</strong>`), `missing workflow title: ${title}`);
    assert.ok(html.includes(`<small>${detail}</small>`), `missing workflow detail: ${detail}`);
  }

  assert.doesNotMatch(html, /输出、部署与持续迭代/);
  const oldDetails = [
    'yaml · python · structure',
    'dataset · registry · contract',
    'Trainer · checkpoint · metric',
    'Predictor · preprocess · inference',
    'file · ASE · downstream',
    'csv · artifact · workflow',
  ];
  for (const detail of oldDetails) {
    assert.ok(!html.includes(`<small>${detail}</small>`), `obsolete workflow detail remains: ${detail}`);
  }
});

test('all documented task datasets are represented on capability cards', () => {
  const labels = [
    'MP2018 / 2020 / 2024',
    'JARVIS dft_2d / dft_3d',
    'Alexandria pbe_2d',
    'ML2DDB',
    'ALEX MP20',
    'MPtrj',
    'MP_EC',
    'MD17_EC',
    'QM9_EC',
    'OMol25_EC',
    'MSD-NMR',
    'HAADF STEM',
    'BF STEM',
  ];
  for (const label of labels) assert.ok(html.includes(label), `missing dataset label: ${label}`);
  assert.match(html, /<h3>机器学习原子间势函数<\/h3>/);
  assert.doesNotMatch(html, /机器学习原子势/);
  assert.match(html, /<h3>属性预测<\/h3>[\s\S]*?<span>SphereNet<\/span>/);
  assert.match(html, /<h3>机器学习原子间势函数<\/h3>[\s\S]*?<span>SphereNet<\/span>/);
  assert.doesNotMatch(html, /n&lt;15 \/ 20 \/ 25 \/ 35/);
});

test('quickstart uses complete README training and inference commands', () => {
  assert.ok(html.includes(trainCommand), 'initial training command is incomplete');
  const inferenceCommand = "python property_prediction/predict.py --model_name='megnet_mp2018_train_60k_e_form' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/' --save_path='result.csv'";
  assert.ok(script.includes(inferenceCommand), 'inference command should be represented as one copyable line');
  for (const fragment of [
    'property_prediction/predict.py',
    "--model_name='megnet_mp2018_train_60k_e_form'",
    "--weights_name='best.pdparams'",
    "--cif_file_path='./property_prediction/example_data/cifs/'",
    "--save_path='result.csv'",
  ]) {
    assert.ok(script.includes(fragment), `missing inference command fragment: ${fragment}`);
  }
  assert.doesNotMatch(script, /command\.slice\(6\)/, 'command rendering must not rely on slicing a multiline command');
  assert.doesNotMatch(script, /property_prediction\/predict\.py\s*\\\s*\n/, 'inference display should not split the command with line continuations');
  assert.match(script, /html: String\.raw`<span class=\"token-purple\">python<\/span> property_prediction\/predict\.py/, 'inference display should keep syntax highlighting');
});


test('GitHub Pages homepage is published from docs without a separate assets directory', () => {
  const docsDir = homepageDir;
  const docsIndex = path.join(docsDir, 'index.html');
  assert.ok(existsSync(docsIndex), 'docs/index.html must be the GitHub Pages entry point');
  const publishedHtml = readFileSync(docsIndex, 'utf8');
  assert.match(publishedHtml, /src="\.\/ppmat_logo_image\.png"/);
  assert.match(publishedHtml, /src="\.\/ppmat_logo_character\.png"/);
  assert.match(publishedHtml, /src="\.\/materials-discovery-loop\.png"/);
  assert.doesNotMatch(publishedHtml, /(?:src|href)="\.\/assets\//);
  assert.equal(existsSync(path.join(__dirname, '..', 'homepage', 'assets')), false, 'homepage/assets should be removed');
});


test('README links to the published homepage badge', () => {
  assert.match(
    readme,
    /<a href="https:\/\/paddlepaddle\.github\.io\/PaddleMaterials\/">\s*<img alt="Homepage" src="https:\/\/img\.shields\.io\/badge\/Homepage-PaddleMaterials-[^"]+">\s*<\/a>/,
  );
});
