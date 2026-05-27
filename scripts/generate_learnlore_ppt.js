const pptxgen = require('/tmp/opencode/node_modules/pptxgenjs');

const pptx = new pptxgen();
pptx.defineLayout({ name: 'LEARNLORE_4X3', width: 10, height: 7.5 });
pptx.layout = 'LEARNLORE_4X3';
pptx.author = 'Syed Armaan Ali';
pptx.subject = 'LearnLore major project presentation';
pptx.title = 'LearnLore Project Presentation';
pptx.company = 'Jamia Hamdard';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Georgia',
  bodyFontFace: 'Aptos',
  lang: 'en-US',
};

const C = {
  black: '222222',
  dark: '211D1D',
  gray: '4D4D4D',
  muted: '8A8A8A',
  card: 'FFF7F5',
  card2: 'F8F8F8',
  line: 'E7CFC6',
  orange: 'FF7F2A',
  blue: '2D609B',
  green: '16895B',
  red: 'B43A32',
  purple: '7B4AA0',
  cream: 'FBF7F3',
  white: 'FFFFFF',
};

const W = 10;
const H = 7.5;
const titleX = 0.82;
const leftStripeX = 0.08;

function addFrame(slide, dark = false, slideNo = '') {
  slide.background = { color: dark ? C.dark : C.white };
  const stripeColors = dark ? [C.red, C.line, C.green, C.orange] : [C.line, 'B8D9E8', C.line, C.orange];
  [0, 1, 2, 3].forEach((i) => {
    slide.addShape(pptx.ShapeType.rect, {
      x: leftStripeX + i * 0.06,
      y: 0,
      w: 0.018,
      h: H,
      fill: { color: stripeColors[i] },
      line: { color: stripeColors[i] },
    });
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: W - 0.06,
    y: 0,
    w: 0.018,
    h: H,
    fill: { color: dark ? C.red : C.line },
    line: { color: dark ? C.red : C.line },
  });
  if (slideNo) {
    slide.addText(`LearnLore | ${slideNo}`, {
      x: 0.78,
      y: 7.18,
      w: 1.8,
      h: 0.18,
      fontFace: 'Aptos',
      fontSize: 8,
      color: dark ? 'A8A8A8' : C.muted,
      margin: 0,
    });
  }
}

function addTitle(slide, title, subtitle, dark = false) {
  slide.addText(title, {
    x: titleX,
    y: 0.38,
    w: 8.7,
    h: 0.55,
    fontFace: 'Georgia',
    fontSize: 34,
    bold: true,
    color: dark ? C.white : C.black,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: titleX + 0.05,
      y: 1.08,
      w: 8.5,
      h: 0.25,
      fontFace: 'Aptos',
      fontSize: 13,
      color: dark ? 'D0D0D0' : C.gray,
      margin: 0,
    });
  }
}

function addCard(slide, x, y, w, h, heading, body, accent = C.orange, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.12,
    fill: { color: opts.fill || C.card },
    line: { color: opts.line || C.line, width: 1.1 },
    shadow: opts.shadow === false ? undefined : { type: 'outer', color: '777777', opacity: 0.18, blur: 2, angle: 45, offset: 1.5 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x, y: y + 0.01, w: 0.08, h: h - 0.02,
    fill: { color: accent },
    line: { color: accent },
  });
  slide.addText(heading, {
    x: x + 0.22,
    y: y + 0.17,
    w: w - 0.35,
    h: 0.28,
    fontFace: 'Aptos',
    fontSize: opts.headingSize || 14,
    bold: true,
    color: opts.darkText || C.black,
    margin: 0,
    fit: 'shrink',
  });
  slide.addText(body, {
    x: x + 0.22,
    y: y + 0.58,
    w: w - 0.35,
    h: h - 0.72,
    fontFace: 'Aptos',
    fontSize: opts.bodySize || 12.2,
    color: opts.bodyColor || C.gray,
    margin: 0,
    breakLine: false,
    fit: 'shrink',
  });
}

function addBulletList(slide, items, x, y, w, h, fontSize = 16, dark = false) {
  const runs = [];
  items.forEach((item, idx) => {
    runs.push({ text: item, options: { breakLine: idx < items.length - 1 } });
  });
  slide.addText(runs, {
    x, y, w, h,
    fontFace: 'Aptos',
    fontSize,
    color: dark ? C.white : C.black,
    margin: 0,
    breakLine: false,
    fit: 'shrink',
    paraSpaceAfterPt: 10,
  });
}

function addFooterNote(slide, text, dark = false) {
  slide.addText(text, {
    x: 1.05,
    y: 6.45,
    w: 8.35,
    h: 0.55,
    fontFace: 'Aptos',
    fontSize: 16,
    bold: true,
    color: dark ? C.white : C.black,
    align: 'center',
    margin: 0,
    fit: 'shrink',
  });
}

function addCircleStat(slide, x, y, color, value, label) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: 1.18, h: 1.18,
    fill: { color },
    line: { color },
    shadow: { type: 'outer', color: '777777', opacity: 0.25, blur: 2, angle: 45, offset: 1.5 },
  });
  slide.addText(value, {
    x, y: y + 0.35, w: 1.18, h: 0.3,
    fontFace: 'Aptos', fontSize: 20, bold: true,
    color: C.white, align: 'center', margin: 0,
  });
  slide.addText(label, {
    x: x - 0.3, y: y + 1.34, w: 1.8, h: 0.32,
    fontFace: 'Aptos', fontSize: 10.5,
    color: C.gray, align: 'center', margin: 0,
    fit: 'shrink',
  });
}

function addFlowBox(slide, x, y, w, h, text, fill = C.card, color = C.black) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill },
    line: { color: C.line, width: 1 },
    shadow: { type: 'outer', color: '777777', opacity: 0.18, blur: 1.5, angle: 45, offset: 1.2 },
  });
  slide.addText(text, {
    x: x + 0.08, y: y + 0.16, w: w - 0.16, h: h - 0.25,
    fontFace: 'Aptos', fontSize: 10.5, bold: true,
    color, align: 'center', valign: 'mid', margin: 0,
    fit: 'shrink',
  });
}

function addArrow(slide, x, y, w, h, color = C.orange) {
  slide.addShape(pptx.ShapeType.rightArrow, {
    x, y, w, h,
    fill: { color },
    line: { color },
    shadow: { type: 'outer', color: '777777', opacity: 0.2, blur: 1, angle: 45, offset: 1 },
  });
}

// 1. Title
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '01');
  slide.addText('A Project Report\nPRESENTATION', {
    x: 3.4, y: 0.55, w: 3.6, h: 0.9,
    fontFace: 'Georgia', fontSize: 28, bold: true,
    align: 'center', color: C.black, margin: 0,
  });
  slide.addText('LearnLore - AI-Powered\nStory-Based Learning Platform', {
    x: 2.45, y: 2.18, w: 5.2, h: 0.8,
    fontFace: 'Georgia', fontSize: 25, bold: true, italic: true,
    align: 'center', color: C.black, margin: 0,
    fit: 'shrink',
  });
  slide.addText('SUBMITTED BY', { x: 1.45, y: 4.0, w: 2.5, h: 0.3, fontFace: 'Georgia', fontSize: 20, bold: true, italic: true, margin: 0 });
  slide.addText('Syed Armaan Ali\nEnrollment No.: 2022-310-221\nB.Tech CSE', { x: 1.35, y: 4.65, w: 3.75, h: 1.2, fontFace: 'Georgia', fontSize: 15.4, bold: true, margin: 0, breakLine: false, fit: 'shrink' });
  slide.addShape(pptx.ShapeType.ellipse, { x: 4.48, y: 4.05, w: 1.0, h: 1.0, fill: { color: 'EEF8EF' }, line: { color: C.green, width: 1.5 } });
  slide.addText('JH', { x: 4.48, y: 4.36, w: 1.0, h: 0.32, fontFace: 'Georgia', fontSize: 18, bold: true, color: C.green, align: 'center', margin: 0 });
  slide.addText('SUPERVISED BY', { x: 6.3, y: 4.0, w: 2.7, h: 0.3, fontFace: 'Georgia', fontSize: 20, bold: true, italic: true, margin: 0 });
  slide.addText('Ms. Juveria\nSupervisor', { x: 6.3, y: 4.65, w: 2.7, h: 0.65, fontFace: 'Georgia', fontSize: 17, bold: true, margin: 0 });
  slide.addText('SUBMITTED TO', { x: 4.1, y: 6.05, w: 2.3, h: 0.28, fontFace: 'Georgia', fontSize: 17, bold: true, italic: true, align: 'center', margin: 0 });
  slide.addText('Department of Computer Science & Engineering\nSchool of Engineering Sciences & Technology\nJAMIA HAMDARD', { x: 2.55, y: 6.55, w: 5.5, h: 0.7, fontFace: 'Georgia', fontSize: 13, bold: true, align: 'center', margin: 0, fit: 'shrink' });
}

// 2. Objective
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '02');
  addTitle(slide, 'Objective', 'What the project set out to achieve');
  addCard(slide, 0.85, 1.72, 2.55, 1.18, 'Story-Based Recall', 'Convert dense study material into concise narratives that improve long-term retention.', C.orange);
  addCard(slide, 3.65, 1.72, 2.55, 1.18, 'Structured AI Output', 'Generate a title, emoji, single story, and memory hooks in a deterministic format.', C.blue);
  addCard(slide, 6.45, 1.72, 2.55, 1.18, 'Modern Study Workflow', 'Support topic input, file upload, flashcards, login, sessions, and future progress tracking.', C.green);
  addBulletList(slide, [
    '- Accept typed topics or supported files such as PDF, TXT, MD, PNG, JPG, and WEBP.',
    '- Validate uploads, normalize source material, and prepare safe prompt context.',
    '- Provide a scalable foundation for document libraries, spaced repetition, and dashboards.'
  ], 1.15, 3.65, 8.1, 1.15, 14.2);
}

// 3. Abstract
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '03');
  addTitle(slide, 'Abstract', 'From passive notes to memorable AI-assisted learning');
  addBulletList(slide, [
    'LearnLore addresses the recall gap students face when revising large volumes of abstract academic content.',
    'The system transforms a topic or uploaded first-page source material into a story and targeted memory hooks.',
    'Its backend combines validation, upload normalization, prompt control, AI polling, and fallback parsing into one reliable pipeline.'
  ], 1.0, 1.75, 5.6, 3.9, 16.5);
  addCircleStat(slide, 7.0, 1.55, C.orange, '8 MB', 'upload limit');
  addCircleStat(slide, 7.85, 3.15, C.blue, '6', 'supported file types');
  addCircleStat(slide, 7.0, 4.75, C.green, '4-6', 'memory hooks');
}

// 4. Problem Statement
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '04');
  addTitle(slide, 'Problem Statement', 'Why ordinary study notes are not enough');
  addCard(slide, 0.92, 1.72, 2.45, 1.88, 'Recall Gap', 'Students have access to notes, but static notes rarely become durable memory cues.', C.orange, { bodySize: 12.5 });
  addCard(slide, 3.72, 1.72, 2.45, 1.88, 'AI Reliability Risk', 'Generic AI explanations may be verbose, unstructured, or unsupported by source material.', C.red, { bodySize: 12.5 });
  addCard(slide, 6.52, 1.72, 2.45, 1.88, 'Input Complexity', 'PDF pages, screenshots, and text files need safe preprocessing before generation.', C.green, { bodySize: 12.5 });
  slide.addText('Central challenge: build a web-based educational system that accepts concise topics or limited source material, validates it safely, and returns memorable story-based explanations with targeted hooks while remaining structured, fast, and extensible.', {
    x: 1.2, y: 4.55, w: 7.9, h: 1.12,
    fontFace: 'Aptos', fontSize: 19, bold: true,
    align: 'center', color: C.black, margin: 0,
    fit: 'shrink',
  });
}

// 5. Latest Literature Review
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '05');
  addTitle(slide, 'Latest Literature Review', 'Key technical foundations used in this work');
  const cards = [
    ['Next.js App Router', 'Route handlers, server actions, and browser-facing React workflows for full-stack development.', C.orange],
    ['React 19', 'Component-based UI for story generation, flashcard review, and quiz interaction.', C.blue],
    ['Prisma + PostgreSQL', 'Schema-driven models for users, sessions, documents, concepts, stories, and progress.', C.green],
    ['AI Prompt Engineering', 'Constrained prompt templates reduce hallucination and make response parsing predictable.', C.red],
    ['Active Recall', 'Flashcards and quizzes reinforce memory through retrieval practice after story-based explanation.', C.orange],
    ['Secure Auth Design', 'Argon2 password hashing, HTTP-only cookies, bearer tokens, and session validation.', C.blue],
  ];
  cards.forEach((c, i) => addCard(slide, 0.95 + (i % 2) * 4.25, 1.55 + Math.floor(i / 2) * 1.38, 3.78, 0.98, c[0], c[1], c[2], { bodySize: 11.6, headingSize: 13.2 }));
}

// 6. Present Investigation
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '06');
  addTitle(slide, 'Present Investigation', 'What was built and tested');
  const y = 1.85;
  const xs = [0.82, 2.42, 4.02, 5.62, 7.22];
  ['User Input', 'Validation', 'Upload\nNormalization', 'AI Generation', 'Story + Hooks'].forEach((t, i) => {
    addFlowBox(slide, xs[i], y, 1.25, 0.72, t);
    if (i < 4) addArrow(slide, xs[i] + 1.26, y + 0.26, 0.5, 0.18);
  });
  addBulletList(slide, [
    '- Main generation route accepts multipart form data with a topic and optional file.',
    '- PDF, image, and text inputs are normalized differently before prompt construction.',
    '- AI responses are parsed into application data with fallback title, emoji, story, and hook handling.',
    '- Authentication and sessions support future personalized learning history.'
  ], 1.0, 3.22, 3.95, 1.95, 13.2);
  addBulletList(slide, [
    '- Flashcard and quiz modules extend story reading into active recall practice.',
    '- Prisma schema anticipates persistence for documents, concepts, stories, and progress.',
    '- White-box and black-box test cases cover validation, auth, parsing, and generation flows.'
  ], 5.4, 3.22, 3.9, 1.95, 13.2);
}

// 7. Proposed Solution
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '07');
  addTitle(slide, 'Proposed Solution', 'LearnLore architecture and controlled generation strategy');
  const top = [
    ['Input Layer', 'Topic box, file picker, recent concepts, flashcards, and quizzes.', C.orange],
    ['API Layer', 'Validates input, prepares sources, invokes AI, and parses results.', C.blue],
    ['Learning Output', 'Story, hooks, flashcards, review grades, and mastery progress.', C.green],
    ['Auth & Sessions', 'Register, login, logout, secure cookies, and bearer sessions.', C.red],
    ['Persistence Model', 'Prisma entities for users, documents, concepts, stories, and progress.', C.orange],
    ['Defensive Parsing', 'Fallbacks protect the UI when AI formatting varies.', C.blue],
  ];
  top.forEach((c, i) => addCard(slide, 0.85 + (i % 3) * 3.0, 1.48 + Math.floor(i / 3) * 1.42, 2.55, 0.98, c[0], c[1], c[2], { bodySize: 11.2, headingSize: 12.4 }));
  slide.addText('Architecture view', { x: 1.0, y: 4.42, w: 2.0, h: 0.25, fontFace: 'Aptos', fontSize: 12, bold: true, color: C.gray, margin: 0 });
  const y = 5.0;
  const boxes = ['Topic/File', 'Validate\n& Normalize', 'Build Prompt', 'Aye Chat\n+ Poll', 'Parse Concept', 'Display\nReview'];
  boxes.forEach((b, i) => {
    addFlowBox(slide, 0.78 + i * 1.53, y, 1.18, 0.62, b, i === 3 ? C.blue : C.card2, i === 3 ? C.white : C.black);
    if (i < boxes.length - 1) addArrow(slide, 1.98 + i * 1.53, y + 0.24, 0.32, 0.14);
  });
}

// 8. Validation Tools & Datasets
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '08');
  addTitle(slide, 'Validation Tools & Datasets', 'Experimental basis used for evaluation');
  const x = 0.95, y = 1.55, rowH = 0.55;
  slide.addShape(pptx.ShapeType.rect, { x, y, w: 8.25, h: rowH, fill: { color: C.black }, line: { color: C.black } });
  slide.addText('Category', { x: x + 0.1, y: y + 0.14, w: 1.7, h: 0.2, fontSize: 12.5, fontFace: 'Aptos', bold: true, color: C.white, margin: 0 });
  slide.addText('Details', { x: x + 2.05, y: y + 0.14, w: 5.8, h: 0.2, fontSize: 12.5, fontFace: 'Aptos', bold: true, color: C.white, margin: 0 });
  const rows = [
    ['Input dataset', 'Student-provided topic, PDF first page, text/markdown notes, or image snapshot'],
    ['Frameworks', 'Next.js 16, React 19, TypeScript, Prisma, PostgreSQL-oriented schema'],
    ['AI service', 'Aye Chat integration with token-based invocation and polling completion'],
    ['Validation checks', 'Required input, 8 MB size limit, supported file type, token shape, auth uniqueness'],
    ['Testing methods', 'White-box paths for session/parser logic; black-box cases for generation and registration'],
  ];
  rows.forEach((r, i) => {
    const yy = y + rowH + i * 0.58;
    slide.addShape(pptx.ShapeType.rect, { x, y: yy, w: 8.25, h: 0.52, fill: { color: i % 2 === 0 ? C.card : C.white }, line: { color: i % 2 === 0 ? C.card : C.white } });
    slide.addText(r[0], { x: x + 0.1, y: yy + 0.15, w: 1.72, h: 0.18, fontSize: 11.2, fontFace: 'Aptos', bold: true, color: C.black, margin: 0, fit: 'shrink' });
    slide.addText(r[1], { x: x + 2.05, y: yy + 0.15, w: 5.8, h: 0.18, fontSize: 10.8, fontFace: 'Aptos', color: C.black, margin: 0, fit: 'shrink' });
  });
  addFooterNote(slide, 'Main validation principle: reject unsafe input early, then parse AI output defensively before presenting it to the learner.');
}

// 9. Empirical Evaluation
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '09');
  addTitle(slide, 'Empirical Evaluation', 'Prototype behavior and software-quality evaluation');
  addCard(slide, 0.95, 1.45, 3.7, 0.86, 'Generation flow', 'Topic/file requests return a structured concept object.', C.orange, { bodySize: 11.8 });
  addCard(slide, 5.05, 1.45, 3.7, 0.86, 'Auth flow', 'Registration, login, logout, and /auth/me use secure sessions.', C.green, { bodySize: 11.8 });
  addCard(slide, 0.95, 2.52, 3.7, 0.86, 'Failure handling', 'Invalid input and upload errors return explicit messages.', C.blue, { bodySize: 11.8 });
  addCard(slide, 5.05, 2.52, 3.7, 0.86, 'Review modules', 'Flashcards and quizzes add active recall practice.', C.red, { bodySize: 11.8 });
  slide.addText('Validation coverage by area', { x: 1.05, y: 3.85, w: 2.6, h: 0.25, fontFace: 'Aptos', fontSize: 15, bold: true, margin: 0 });
  const bars = [
    ['Input validation', 0.95, C.orange], ['Upload handling', 0.8, C.blue], ['Auth/session', 0.9, C.green], ['AI parsing', 0.75, C.red], ['Future analytics', 0.35, C.purple]
  ];
  bars.forEach((b, i) => {
    const yy = 4.35 + i * 0.42;
    slide.addText(b[0], { x: 1.1, y: yy + 0.03, w: 1.8, h: 0.18, fontSize: 10.8, fontFace: 'Aptos', margin: 0 });
    slide.addShape(pptx.ShapeType.rect, { x: 3.0, y: yy, w: 3.6, h: 0.18, fill: { color: 'E7E7E7' }, line: { color: 'E7E7E7' }, shadow: { type: 'outer', color: '777777', opacity: 0.16, blur: 1, angle: 45, offset: 1 } });
    slide.addShape(pptx.ShapeType.rect, { x: 3.0, y: yy, w: 3.6 * b[1], h: 0.18, fill: { color: b[2] }, line: { color: b[2] } });
    slide.addText(`${Math.round(b[1] * 100)}%`, { x: 6.78, y: yy - 0.02, w: 0.55, h: 0.18, fontSize: 9.5, bold: true, fontFace: 'Aptos', margin: 0 });
  });
  const tx = 7.22, ty = 3.85;
  slide.addShape(pptx.ShapeType.rect, { x: tx, y: ty, w: 1.8, h: 0.36, fill: { color: C.black }, line: { color: C.black } });
  slide.addText('Metric', { x: tx + 0.08, y: ty + 0.1, w: 0.7, h: 0.14, fontSize: 8.8, color: C.white, bold: true, margin: 0 });
  slide.addText('Result', { x: tx + 1.0, y: ty + 0.1, w: 0.7, h: 0.14, fontSize: 8.8, color: C.white, bold: true, margin: 0 });
  [['FRs', '12'], ['NFRs', '7'], ['UCs', '8'], ['Tests', '15']].forEach((r, i) => {
    const yy = ty + 0.36 + i * 0.36;
    slide.addShape(pptx.ShapeType.rect, { x: tx, y: yy, w: 1.8, h: 0.34, fill: { color: i % 2 ? C.white : C.card }, line: { color: i % 2 ? C.white : C.card } });
    slide.addText(r[0], { x: tx + 0.08, y: yy + 0.1, w: 0.7, h: 0.12, fontSize: 8.6, margin: 0 });
    slide.addText(r[1], { x: tx + 1.1, y: yy + 0.1, w: 0.5, h: 0.12, fontSize: 8.6, bold: true, margin: 0 });
  });
  addFooterNote(slide, 'Prototype evaluation emphasizes stable validation, structured generation, auth/session safety, and a clear path to progress analytics.');
}

// 10. Conclusion
{
  const slide = pptx.addSlide();
  addFrame(slide, true, '10');
  addTitle(slide, 'Conclusion', 'Main outcomes of the project', true);
  const cardOpts = { fill: '383432', line: C.orange, shadow: true, darkText: C.orange, bodyColor: C.white, headingSize: 16, bodySize: 13 };
  addCard(slide, 0.95, 2.0, 2.35, 2.15, 'Controlled\nAI Pipeline', 'Validated input, normalized files, constrained prompts, polling, and structured parsing.', C.orange, cardOpts);
  addCard(slide, 3.7, 2.0, 2.35, 2.15, 'Useful\nStudy Output', 'Narrative explanations and memory hooks make revision more engaging and memorable.', C.green, { ...cardOpts, line: C.green, darkText: '5FE0A8' });
  addCard(slide, 6.45, 2.0, 2.35, 2.15, 'Extensible\nPlatform', 'Authentication, sessions, flashcards, quizzes, and schema entities support future growth.', C.blue, { ...cardOpts, line: C.blue, darkText: '9FD0FF' });
  slide.addText('Implemented Core', { x: 1.15, y: 4.65, w: 2.0, h: 0.28, fontFace: 'Aptos', fontSize: 15, bold: true, color: C.orange, align: 'center', margin: 0 });
  slide.addText('Learner Value', { x: 4.0, y: 4.65, w: 1.75, h: 0.28, fontFace: 'Aptos', fontSize: 15, bold: true, color: '5FE0A8', align: 'center', margin: 0 });
  slide.addText('Growth Path', { x: 6.9, y: 4.65, w: 1.5, h: 0.28, fontFace: 'Aptos', fontSize: 15, bold: true, color: '9FD0FF', align: 'center', margin: 0 });
  addFooterNote(slide, 'LearnLore shows that educational AI quality depends on validation, response discipline, and learner-focused UI design as much as on the generation model itself.', true);
}

// 11. Limitations & Future Scope
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '11');
  addTitle(slide, 'Limitations & Future Scope', 'What remains beyond the current study');
  addCard(slide, 0.9, 1.55, 3.9, 1.05, 'Limitations', 'Current PDF processing uses only the first page; generated quality depends on input clarity and external AI availability.', C.red, { bodySize: 12.5 });
  addCard(slide, 5.15, 1.55, 3.9, 1.05, 'Immediate Future', 'Persist all generated concepts, stories, hooks, decks, and mastery records against user accounts.', C.green, { bodySize: 12.5 });
  addBulletList(slide, [
    '- Add multi-page PDF ingestion with chunking and concept segmentation.',
    '- Improve OCR and handwriting support for photographed class notes.',
    '- Introduce spaced repetition scheduling for flashcard review.'
  ], 1.0, 3.25, 3.9, 1.45, 14);
  addBulletList(slide, [
    '- Build topic-wise progress dashboards using UserProgress.',
    '- Add teacher/supervisor reporting with privacy-safe aggregate analytics.',
    '- Support multilingual generation and exportable revision sheets.'
  ], 5.25, 3.25, 3.85, 1.45, 14);
  addFooterNote(slide, 'The present architecture already separates auth, upload normalization, AI invocation, and persistence-ready data models.');
}

// 12. References
{
  const slide = pptx.addSlide();
  addFrame(slide, false, '12');
  addTitle(slide, 'References', 'IEEE-style sources used in the report');
  addCard(slide, 0.98, 1.6, 3.75, 1.15, '[1] Next.js / React', 'Next.js App Router and Route Handlers documentation; React 19 reference documentation, accessed May 2026.', C.orange, { bodySize: 10.5 });
  addCard(slide, 5.1, 1.6, 3.75, 1.15, '[2] Database & ORM', 'Prisma ORM and PostgreSQL documentation for schema design, relations, and persistence model.', C.blue, { bodySize: 10.5 });
  addCard(slide, 0.98, 3.35, 3.75, 1.15, '[3] Security & Processing', 'Argon2 hashing references; PDF parsing and upload-handling package documentation.', C.green, { bodySize: 10.5 });
  addCard(slide, 5.1, 3.35, 3.75, 1.15, '[4] Project Artifacts', 'LearnLore repository source files including generate route, uploads, Aye Chat client, Prisma schema, auth routes, and final report.', C.red, { bodySize: 10.5 });
  slide.addText('Thank You', { x: 4.25, y: 5.62, w: 1.6, h: 0.35, fontFace: 'Aptos', fontSize: 18, bold: true, align: 'center', margin: 0 });
}

pptx.writeFile({ fileName: 'report/LearnLore_Project_Presentation.pptx' });
