const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Shahbaz Ali';
pptx.company = 'Jamia Hamdard University';
pptx.subject = 'A Joint Learning Framework for Universal Sound Separation and Classification';
pptx.title = 'Universal Sound Separation and Classification';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Aptos Display',
  bodyFontFace: 'Aptos',
  lang: 'en-US'
};

const W = 13.333;
const H = 7.5;
const C = {
  navy: '18263B',
  blue: '2A6F97',
  teal: '00A896',
  mint: 'CFF7E8',
  sand: 'F6F1E8',
  paper: 'FCFBF8',
  ink: '1D2433',
  muted: '64748B',
  orange: 'F29E4C',
  red: 'C94C4C',
  green: '2D936C',
  white: 'FFFFFF',
  paleBlue: 'E8F3F8',
  paleOrange: 'FFF0DF'
};

function addBg(slide, color = C.paper) {
  slide.background = { color };
}

function addSection(slide, label, idx) {
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.18, fill: { color: C.teal }, line: { color: C.teal } });
  slide.addText(label, { x: 0.55, y: 0.35, w: 8.6, h: 0.25, fontFace: 'Aptos', fontSize: 8.5, bold: true, color: C.teal, charSpacing: 1.2, margin: 0 });
  slide.addText(String(idx).padStart(2, '0'), { x: 12.25, y: 0.3, w: 0.55, h: 0.35, fontSize: 11, bold: true, color: C.muted, align: 'right', margin: 0 });
}

function title(slide, text, subtitle, idx) {
  addBg(slide);
  addSection(slide, subtitle, idx);
  slide.addText(text, { x: 0.55, y: 0.72, w: 8.2, h: 0.55, fontFace: 'Aptos Display', fontSize: 29, bold: true, color: C.ink, margin: 0, fit: 'shrink' });
}

function bulletList(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  items.forEach((item, i) => {
    runs.push({ text: item, options: { bullet: { indent: 14 }, breakLine: i !== items.length - 1 } });
  });
  slide.addText(runs, {
    x, y, w, h,
    fontSize: opts.fontSize || 15,
    color: opts.color || C.ink,
    breakLine: false,
    fit: 'shrink',
    paraSpaceAfterPt: opts.paraSpaceAfterPt || 7,
    margin: 0.04,
    valign: 'top'
  });
}

function pill(slide, txt, x, y, w, color = C.teal, textColor = C.white) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.36, rectRadius: 0.12, fill: { color }, line: { color } });
  slide.addText(txt, { x: x + 0.08, y: y + 0.09, w: w - 0.16, h: 0.16, fontSize: 8.5, bold: true, color: textColor, align: 'center', margin: 0, fit: 'shrink' });
}

function card(slide, x, y, w, h, header, body, color = C.white) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color }, line: { color: 'D7DEE8', width: 0.8 }, shadow: { type: 'outer', color: '718096', opacity: 0.14, blur: 1, offset: 1, angle: 45 } });
  slide.addText(header, { x: x + 0.22, y: y + 0.18, w: w - 0.44, h: 0.26, fontSize: 14, bold: true, color: C.ink, margin: 0, fit: 'shrink' });
  slide.addText(body, { x: x + 0.22, y: y + 0.58, w: w - 0.44, h: h - 0.76, fontSize: 11.8, color: C.muted, margin: 0.02, fit: 'shrink', breakLine: false, valign: 'top' });
}

function metric(slide, x, y, val, label, color = C.teal, labelColor = C.muted) {
  slide.addText(val, { x, y, w: 2.2, h: 0.52, fontSize: 30, bold: true, color, margin: 0, align: 'center' });
  slide.addText(label, { x, y: y + 0.58, w: 2.2, h: 0.36, fontSize: 10.5, color: labelColor, bold: true, align: 'center', margin: 0, fit: 'shrink' });
}

function addFooter(slide) {
  slide.addText('A Joint Learning Framework for Universal Sound Separation and Classification', { x: 0.55, y: 7.12, w: 8.3, h: 0.18, fontSize: 7.5, color: '8A97A8', margin: 0 });
}

function addMiniWave(slide, x, y, w, color = C.teal) {
  const step = w / 14;
  for (let i = 0; i < 14; i++) {
    const amp = [0.18,0.34,0.56,0.25,0.72,0.4,0.6,0.3,0.5,0.7,0.35,0.52,0.24,0.36][i];
    slide.addShape(pptx.ShapeType.rect, { x: x + i * step, y: y + (0.76 - amp) / 2, w: step * 0.42, h: amp, fill: { color, transparency: 4 }, line: { color } });
  }
}

// 1. Title
{
  const s = pptx.addSlide();
  addBg(s, C.navy);
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: H, fill: { color: C.navy }, line: { color: C.navy } });
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.5, h: H, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText('A Joint Learning Framework for Universal\nSound Separation and Classification', { x: 0.9, y: 1.05, w: 8.9, h: 1.28, fontFace: 'Aptos Display', fontSize: 31, bold: true, color: C.white, margin: 0, fit: 'shrink' });
  s.addText('Research-based B.Tech project presentation', { x: 0.92, y: 2.58, w: 5.3, h: 0.28, fontSize: 15, color: C.mint, margin: 0 });
  addMiniWave(s, 9.95, 1.05, 2.45, C.teal);
  ['Universal Sound Separation', 'Dual-Decoder Architecture', 'Source Counting'].forEach((t, i) => pill(s, t, 0.92 + i * 2.5, 3.18, 2.25, i === 1 ? C.orange : C.teal));
  s.addShape(pptx.ShapeType.roundRect, { x: 0.9, y: 4.55, w: 5.5, h: 1.45, rectRadius: 0.12, fill: { color: '22344D' }, line: { color: '36506E' } });
  s.addText('Student: Shahbaz Ali\nEnrollment No.: 2022-310-209\nSupervisor: Dr. Hira Javed', { x: 1.18, y: 4.86, w: 4.85, h: 0.78, fontSize: 15, color: C.white, bold: false, margin: 0, breakLine: false, fit: 'shrink' });
  s.addText('Department of Computer Science & Engineering\nSchool of Engineering Sciences & Technology\nJamia Hamdard University, Delhi', { x: 7.15, y: 5.0, w: 4.9, h: 0.8, fontSize: 13, color: 'DDE9F2', margin: 0, align: 'right', fit: 'shrink' });
}

// 2. Objective
{
  const s = pptx.addSlide();
  title(s, 'Objective', 'RESEARCH PRESENTATION', 2);
  s.addText('Upgrade fixed-output universal sound separation into a source-aware system that knows how many sounds are active.', { x: 0.58, y: 1.46, w: 7.35, h: 0.62, fontSize: 18, color: C.ink, bold: true, margin: 0, fit: 'shrink' });
  const objs = [
    ['Separate', 'Extract high-fidelity waveforms from single-channel mixtures.'],
    ['Count', 'Predict the exact active source count from shared latent features.'],
    ['Suppress', 'Zero inactive channels to eliminate hallucinated noise.'],
    ['Stabilize', 'Use two-stage curriculum learning to avoid gradient collapse.']
  ];
  objs.forEach((o, i) => card(s, 0.7 + (i % 2) * 3.55, 2.45 + Math.floor(i / 2) * 1.55, 3.1, 1.18, o[0], o[1], i % 2 ? C.paleBlue : C.white));
  s.addShape(pptx.ShapeType.roundRect, { x: 8.35, y: 1.48, w: 3.85, h: 4.1, rectRadius: 0.18, fill: { color: C.navy }, line: { color: C.navy } });
  metric(s, 8.75, 2.0, '11.2 dB', 'target SI-SNRi achieved', C.mint, 'DDE9F2');
  metric(s, 9.58, 3.15, '+1.7 dB', 'aggregate improvement', C.orange, 'DDE9F2');
  metric(s, 8.75, 4.3, '90.2%', 'source-count accuracy', C.mint, 'DDE9F2');
  addFooter(s);
}

// 3. Abstract
{
  const s = pptx.addSlide();
  title(s, 'Abstract', 'SUMMARY OF WORK', 3);
  card(s, 0.65, 1.45, 3.75, 3.95, 'Challenge', 'Real-world recordings contain overlapping environmental sounds. Fixed-output separators often hallucinate noise into unused channels when fewer sources are active.', C.paleOrange);
  card(s, 4.75, 1.45, 3.75, 3.95, 'Approach', 'A dual-decoder network combines a 32-layer TDCN++ separator with an MLP source-count classifier trained on permutation-invariant energy features.', C.paleBlue);
  card(s, 8.85, 1.45, 3.75, 3.95, 'Outcome', 'Inference-time source zeroing and learned mixture consistency reduce artifact injection, producing strong SI-SNRi gains across variable acoustic mixtures.', C.white);
  addMiniWave(s, 1.15, 5.88, 10.75, C.blue);
  addFooter(s);
}

// 4. Problem Statement
{
  const s = pptx.addSlide();
  title(s, 'Problem Statement', 'WHY FIXED OUTPUT FAILS', 4);
  s.addShape(pptx.ShapeType.roundRect, { x: 0.75, y: 1.42, w: 5.65, h: 4.8, rectRadius: 0.12, fill: { color: C.white }, line: { color: 'D7DEE8' } });
  s.addText('Baseline Fixed-Output TDCN++', { x: 1.05, y: 1.72, w: 3.8, h: 0.28, fontSize: 16, bold: true, color: C.red, margin: 0 });
  ['1-source mixture', '4-channel decoder', 'Clean track', 'Noise', 'Noise', 'Noise'].forEach((t, i) => {
    const coords = [[2.25,2.3],[2.25,3.1],[0.95,4.35],[2.36,4.35],[3.77,4.35],[5.18,4.35]][i];
    s.addShape(pptx.ShapeType.roundRect, { x: coords[0], y: coords[1], w: i < 2 ? 1.95 : 0.94, h: 0.5, rectRadius: 0.06, fill: { color: i >= 3 ? 'FCE6E6' : C.paleBlue }, line: { color: i >= 3 ? C.red : C.blue } });
    s.addText(t, { x: coords[0] + 0.05, y: coords[1] + 0.14, w: i < 2 ? 1.85 : 0.84, h: 0.16, fontSize: i < 2 ? 8.8 : 8.2, color: C.ink, align: 'center', margin: 0, fit: 'shrink' });
  });
  s.addShape(pptx.ShapeType.roundRect, { x: 6.9, y: 1.42, w: 5.65, h: 4.8, rectRadius: 0.12, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Required Source-Aware Behavior', { x: 7.2, y: 1.72, w: 4.1, h: 0.28, fontSize: 16, bold: true, color: C.mint, margin: 0 });
  bulletList(s, [
    'Predict the true active source count K.',
    'Preserve only top-K energy tracks.',
    'Force remaining channels to absolute silence.',
    'Avoid over-separation and artifact injection.'
  ], 7.28, 2.35, 4.6, 2.5, { color: C.white, fontSize: 15 });
  metric(s, 8.8, 4.95, '13.5 dB', 'single-source SI-SNRi after zeroing', C.orange, 'DDE9F2');
  addFooter(s);
}

// 5. Literature Review I
{
  const s = pptx.addSlide();
  title(s, 'Latest Literature Review I', 'FOUNDATIONS', 5);
  const refs = [
    ['Universal Sound Separation', 'Kavalerov et al. defined the FUSS benchmark and exposed over-separation in variable-source mixtures.'],
    ['Conv-TasNet / TDCN', 'Luo and Mesgarani shifted separation from STFT masking to direct time-domain convolution.'],
    ['Permutation Invariant Training', 'Kolbaek et al. resolved arbitrary channel-to-target alignment during multi-source training.'],
    ['Asteroid Toolkit', 'Pariente et al. demonstrated practical PyTorch pipelines for separation research.'],
    ['FSD50K', 'Fonseca et al. provided human-labeled environmental sound events used for source metadata.']
  ];
  refs.forEach((r, i) => card(s, 0.75 + (i % 2) * 6.05, 1.4 + Math.floor(i / 2) * 1.62, i === 4 ? 11.4 : 5.45, 1.18, `${i + 1}. ${r[0]}`, r[1], i === 4 ? C.paleOrange : C.white));
  addFooter(s);
}

// 6. Literature Review II
{
  const s = pptx.addSlide();
  title(s, 'Latest Literature Review II', 'EVOLUTION OF METHODS', 6);
  const timeline = [
    ['ICA', 'Statistical independence; limited by microphone count.'],
    ['NMF', 'Spectral dictionaries; weak generalization.'],
    ['Deep Clustering', 'Embedding bins; costly inference.'],
    ['Deep Attractor Nets', 'Attractor points for masks; still STFT-bound.'],
    ['Time-Domain TDCN++', 'Learns phase-inclusive latent basis; fixed outputs remain a bottleneck.']
  ];
  timeline.forEach((t, i) => {
    const x = 0.8 + i * 2.42;
    s.addShape(pptx.ShapeType.ellipse, { x, y: 2.05, w: 0.56, h: 0.56, fill: { color: i === 4 ? C.orange : C.teal }, line: { color: i === 4 ? C.orange : C.teal } });
    s.addText(['ICA', 'NMF', 'DC', 'DAN', 'TDCN'][i], { x: x + 0.06, y: 2.22, w: 0.44, h: 0.12, fontSize: 6.7, bold: true, color: C.white, align: 'center', margin: 0, fit: 'shrink' });
    if (i < 4) s.addShape(pptx.ShapeType.line, { x: x + 0.62, y: 2.33, w: 1.7, h: 0, line: { color: 'B8C5D3', width: 1.5 } });
    card(s, x - 0.18, 3.0, 2.08, 1.75, t[0], t[1], i === 4 ? C.paleOrange : C.white);
  });
  s.addText('The proposed work builds on time-domain TDCN++ while adding explicit source-count awareness.', { x: 1.1, y: 5.58, w: 10.8, h: 0.45, fontSize: 15.5, color: C.ink, bold: true, align: 'center', margin: 0 });
  addFooter(s);
}

// 7. Present Investigation
{
  const s = pptx.addSlide();
  title(s, 'Present Investigation', 'WHAT WAS EXAMINED', 7);
  card(s, 0.75, 1.45, 3.55, 3.95, 'Architectural Question', 'Can separation quality improve if the model jointly learns source counting and waveform extraction from the same latent representation?', C.white);
  card(s, 4.9, 1.45, 3.55, 3.95, 'Training Question', 'Can curriculum learning prevent multi-task gradient collapse when classification and separation losses share an encoder?', C.paleBlue);
  card(s, 9.05, 1.45, 3.55, 3.95, 'Systems Question', 'Can FIR resampling cache remove CPU starvation and make AWS GPU training viable for 20,000 mixtures?', C.paleOrange);
  ['Architecture', 'Optimization', 'Infrastructure'].forEach((t, i) => pill(s, t, 1.3 + i * 4.15, 5.75, 2.4, i === 2 ? C.orange : C.teal));
  addFooter(s);
}

// 8. Proposed Solution
{
  const s = pptx.addSlide();
  title(s, 'Proposed Solution', 'DUAL-DECODER FRAMEWORK', 8);
  const nodes = [
    ['Input Audio Mixture', 0.95, 2.9, 1.9, C.paleBlue],
    ['Shared Temporal Encoder\n1D Conv + TCN Blocks', 3.25, 2.55, 2.4, C.mint],
    ['Separation Decoder\n4 Estimated Channels', 6.0, 1.65, 2.2, C.white],
    ['MLP Source Counter\nP(N=1..4)', 6.0, 3.75, 2.2, C.paleOrange],
    ['Energy Sort +\nInference-Time Zeroing', 8.65, 2.55, 2.35, C.navy],
    ['Final Clean Sources', 11.2, 2.9, 1.25, C.teal]
  ];
  nodes.forEach(n => {
    s.addShape(pptx.ShapeType.roundRect, { x: n[1], y: n[2], w: n[3], h: 0.8, rectRadius: 0.08, fill: { color: n[4] }, line: { color: n[4] === C.navy ? C.navy : '9FB6C3' } });
    s.addText(n[0], { x: n[1] + 0.1, y: n[2] + 0.19, w: n[3] - 0.2, h: 0.36, fontSize: 11.4, bold: true, color: n[4] === C.navy ? C.white : C.ink, align: 'center', margin: 0, fit: 'shrink' });
  });
  [[2.9,3.3,0.32,0],[5.7,2.95,0.28,-0.9],[5.7,3.35,0.28,0.9],[8.25,2.05,0.38,0.9],[8.25,4.15,0.38,-0.9],[11.04,3.3,0.16,0]].forEach(a => s.addShape(pptx.ShapeType.line, { x: a[0], y: a[1], w: a[2], h: a[3], line: { color: C.blue, width: 2.2, beginArrowType: 'none', endArrowType: 'triangle' } }));
  s.addText('Key mechanism: the classifier predicts K, then the system preserves the top-K energy tracks and zeroes the remaining channels.', { x: 1.05, y: 5.72, w: 11.05, h: 0.4, fontSize: 16, bold: true, color: C.ink, align: 'center', margin: 0 });
  addFooter(s);
}

// 9. Validation Tools / Datasets
{
  const s = pptx.addSlide();
  title(s, 'Validation Tools / Datasets Used', 'EVALUATION SETUP', 9);
  const items = [
    ['FUSS Dataset', '20,000 reverberant mixtures with 1 to 4 active sources.'],
    ['FSD50K Metadata', 'Source annotations used to derive source-count labels.'],
    ['SI-SNR / SI-SNRi', 'Scale-invariant metric for waveform separation quality.'],
    ['Cross-Entropy + Confusion Matrix', 'Classification accuracy and source-count error patterns.'],
    ['AWS G4dn GPU', 'Cloud training environment with optimized data loading.'],
    ['FIR Matrix Caching', 'Pre-computed resampling filters to prevent CPU bottlenecks.']
  ];
  items.forEach((it, i) => card(s, 0.7 + (i % 3) * 4.2, 1.45 + Math.floor(i / 3) * 2.15, 3.55, 1.55, it[0], it[1], i % 2 ? C.white : C.paleBlue));
  addFooter(s);
}

// 10. Empirical evaluation - chart
{
  const s = pptx.addSlide();
  title(s, 'Empirical Evaluation', 'GRAPHICAL COMPARISON', 10);
  s.addChart(pptx.ChartType.bar, [
    { name: 'TDCN++ Baseline', labels: ['N=1', 'N=2', 'N=3', 'N=4', 'Aggregate'], values: [4.2, 9.8, 9.1, 8.7, 9.5] },
    { name: 'Dual-Decoder', labels: ['N=1', 'N=2', 'N=3', 'N=4', 'Aggregate'], values: [13.5, 11.5, 10.7, 8.9, 11.2] }
  ], {
    x: 0.75, y: 1.45, w: 7.45, h: 4.65,
    catAxisLabelFontFace: 'Aptos', catAxisLabelFontSize: 10,
    valAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 14,
    showLegend: true, showValue: true,
    showTitle: false,
    ser: [{ color: '82949D' }, { color: C.blue }],
    valAxisTitle: 'SI-SNRi (dB) - higher is better'
  });
  s.addShape(pptx.ShapeType.roundRect, { x: 8.65, y: 1.55, w: 3.8, h: 4.45, rectRadius: 0.16, fill: { color: C.navy }, line: { color: C.navy } });
  metric(s, 9.45, 2.1, '+9.6 dB', 'single-source improvement', C.orange, 'DDE9F2');
  metric(s, 9.45, 3.45, '+1.7 dB', 'aggregate improvement', C.mint, 'DDE9F2');
  s.addText('Largest gain appears where fixed-output models over-separate: single-source mixtures.', { x: 9.08, y: 5.05, w: 2.95, h: 0.36, fontSize: 12, color: C.white, align: 'center', margin: 0, fit: 'shrink' });
  addFooter(s);
}

// 11. Empirical evaluation - secondary results
{
  const s = pptx.addSlide();
  title(s, 'Empirical Evaluation', 'CLASSIFICATION, TRAINING, SYSTEMS', 11);
  card(s, 0.7, 1.35, 3.6, 4.25, 'Classifier Performance', 'Overall source-count accuracy reached 90.2%. Single-source recall was 95.1%, which is crucial because source zeroing gives the largest quality gain there.', C.white);
  card(s, 4.85, 1.35, 3.6, 4.25, 'Curriculum Impact', 'Training both losses from epoch 1 collapsed to 6.4 dB SI-SNRi. Warmup plus Stage 2 SRT stabilized convergence and reached 11.2 dB.', C.paleBlue);
  card(s, 9.0, 1.35, 3.6, 4.25, 'Data Pipeline Viability', 'FIR caching improved GPU saturation from about 15% to 98%, reducing epoch duration from roughly 4.5 hours to 44 minutes.', C.paleOrange);
  metric(s, 1.4, 4.72, '90.2%', 'count accuracy', C.teal);
  metric(s, 5.55, 4.72, '11.2 dB', 'final SI-SNRi', C.blue);
  metric(s, 9.7, 4.72, '98%', 'GPU saturation', C.orange);
  addFooter(s);
}

// 12. Conclusion
{
  const s = pptx.addSlide();
  addBg(s, C.navy);
  addSection(s, 'CONCLUSION', 12);
  s.addText('CONCLUSION', { x: 0.55, y: 0.35, w: 8.6, h: 0.25, fontFace: 'Aptos', fontSize: 8.5, bold: true, color: C.white, charSpacing: 1.2, margin: 0 });
  s.addText('Conclusion', { x: 0.75, y: 0.9, w: 4.2, h: 0.55, fontSize: 34, bold: true, color: C.white, margin: 0 });
  bulletList(s, [
    'The project resolves over-separation by making the separator source-aware.',
    'Dual-decoder learning combines waveform extraction and source-count classification.',
    'Two-stage curriculum training prevents destructive multi-task gradient interference.',
    'Inference-time source zeroing eliminates hallucinated inactive channels.',
    'Final aggregate performance reaches 11.2 dB SI-SNRi on the FUSS validation setting.'
  ], 0.86, 2.0, 6.8, 3.4, { color: C.white, fontSize: 17, paraSpaceAfterPt: 10 });
  s.addShape(pptx.ShapeType.roundRect, { x: 8.55, y: 1.65, w: 3.8, h: 3.95, rectRadius: 0.18, fill: { color: '22344D' }, line: { color: '36506E' } });
  metric(s, 9.32, 2.3, '11.2 dB', 'SI-SNRi benchmark', C.mint, 'DDE9F2');
  metric(s, 9.32, 3.65, '+1.7 dB', 'over baseline', C.orange, 'DDE9F2');
  addMiniWave(s, 9.22, 5.12, 2.4, C.teal);
}

// 13. Limitations and Future Scope
{
  const s = pptx.addSlide();
  title(s, 'Limitations and Future Scope', 'NEXT DIRECTIONS', 13);
  const futures = [
    ['Scale beyond four sources', 'Extend the mask outputs and classifier space to handle denser acoustic scenes such as traffic intersections.'],
    ['Edge deployment', 'Apply INT8 post-training quantization for IoT devices, hearing aids, and smart-home systems.'],
    ['Real-time streaming', 'Replace non-causal processing with causal dilated convolutions to target sub-10 ms latency.'],
    ['Low-SNR robustness', 'Improve classifier confidence when faint transient events are masked by background noise.']
  ];
  futures.forEach((f, i) => {
    const x = 0.75 + (i % 2) * 6.1;
    const y = 1.45 + Math.floor(i / 2) * 2.15;
    card(s, x, y, 5.45, 1.55, f[0], f[1], i % 2 ? C.paleBlue : C.white);
    s.addShape(pptx.ShapeType.ellipse, { x: x + 4.78, y: y + 0.25, w: 0.42, h: 0.42, fill: { color: i % 2 ? C.blue : C.teal }, line: { color: i % 2 ? C.blue : C.teal } });
  });
  addFooter(s);
}

// 14. References 1
{
  const s = pptx.addSlide();
  title(s, 'References', 'IEEE FORMAT - PART I', 14);
  const refs = [
    '[1] I. Kavalerov et al., "Universal sound separation," WASPAA, 2019.',
    '[2] Y. Luo and N. Mesgarani, "Conv-TasNet," IEEE/ACM TASLP, 2019.',
    '[3] M. Kolbaek et al., "Permutation invariant training," IEEE/ACM TASLP, 2017.',
    '[4] M. Pariente et al., "Asteroid," Proc. Interspeech, 2020.',
    '[5] E. Fonseca et al., "FSD50K," IEEE/ACM TASLP, 2021.',
    '[6] S. Wisdom et al., "Unsupervised sound separation using mixture invariant training," NeurIPS, 2020.',
    '[7] A. Hyvarinen and E. Oja, "Independent component analysis," Neural Networks, 2000.',
    '[8] P. Smaragdis and J. C. Brown, "NMF for polyphonic music transcription," WASPAA, 2003.',
    '[9] J. R. Hershey et al., "Deep clustering," ICASSP, 2016.',
    '[10] J. Chen et al., "Deep attractor network," ICASSP, 2017.'
  ];
  s.addText(refs.slice(0, 5).join('\n'), { x: 0.78, y: 1.08, w: 5.7, h: 5.45, fontSize: 11.4, color: C.ink, margin: 0, fit: 'shrink', breakLine: false, paraSpaceAfterPt: 10 });
  s.addText(refs.slice(5).join('\n'), { x: 6.85, y: 1.08, w: 5.75, h: 5.45, fontSize: 11.4, color: C.ink, margin: 0, fit: 'shrink', breakLine: false, paraSpaceAfterPt: 10 });
  addFooter(s);
}

// 15. References 2
{
  const s = pptx.addSlide();
  title(s, 'References', 'IEEE FORMAT - PART II', 15);
  const refs = [
    '[11] D. Wang and J. Chen, "Supervised speech separation based on deep learning," IEEE/ACM TASLP, 2018.',
    '[12] N. Zeghidour et al., "Wavesplit," IEEE/ACM TASLP, 2021.',
    '[13] K. He et al., "Delving deep into rectifiers," ICCV, 2015.',
    '[14] Y. Bengio et al., "Curriculum learning," ICML, 2009.',
    '[15] E. Tzinis et al., "Improving universal sound separation using sound classification," ICASSP, 2020.',
    '[16] J. O. Smith, "Digital Audio Resampling Home Page," Stanford University, 2007.',
    '[17] P. Micikevicius et al., "Mixed precision training," ICLR, 2018.',
    '[18] E. Nachmani et al., "Voice separation with an unknown number of speakers," ICML, 2020.',
    '[19] F. Pishdadian et al., "Learning to separate sounds with weak supervision," IEEE/ACM TASLP, 2020.',
    '[20] Z.-Q. Wang et al., "Multi-microphone complex spectral mapping," ICASSP, 2020.',
    '[21] G. Mesnil et al., "RNNs for slot filling in spoken language understanding," IEEE/ACM TASLP, 2015.'
  ];
  s.addText(refs.slice(0, 6).join('\n'), { x: 0.78, y: 1.05, w: 5.8, h: 5.65, fontSize: 10.7, color: C.ink, margin: 0, fit: 'shrink', breakLine: false, paraSpaceAfterPt: 8 });
  s.addText(refs.slice(6).join('\n'), { x: 6.85, y: 1.05, w: 5.8, h: 5.65, fontSize: 10.7, color: C.ink, margin: 0, fit: 'shrink', breakLine: false, paraSpaceAfterPt: 8 });
  addFooter(s);
}

pptx.writeFile({ fileName: 'report/universal_sound_separation_presentation.pptx' });
