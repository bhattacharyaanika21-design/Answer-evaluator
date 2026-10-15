/**
 * Builds the project presentation from the latest results.
 *
 *   cd presentation && npm install pptxgenjs react react-dom react-icons sharp
 *   node build_deck.js            ->  AI_Answer_Evaluator.pptx
 *
 * Numbers, charts and worked examples are read from ../results, so re-running
 * train.py + evaluate.py (e.g. with Sentence-BERT) and then this script keeps
 * the slides in sync with the code.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "results");
const OUT = process.argv[2] || path.join(__dirname, "AI_Answer_Evaluator.pptx");

// ------------------------------------------------------------------ data
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') q = false;
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows.filter((r) => r.length > 1);
  return body.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}

const metrics = JSON.parse(fs.readFileSync(path.join(RES, "metrics.json"), "utf8"));
const examples = JSON.parse(fs.readFileSync(path.join(RES, "examples.json"), "utf8"));
const cv = parseCSV(fs.readFileSync(path.join(RES, "cv_predictions.csv"), "utf8"));
const answers = Object.fromEntries(
  parseCSV(fs.readFileSync(path.join(ROOT, "data", "student_answers.csv"), "utf8")).map((r) => [r.answer_id, r]));
const questions = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "questions.json"), "utf8"));

const best = metrics.best_model;
const S = metrics.scoring[best];
const CD = metrics.concept_detection;
const SBERT = metrics.backend === "sbert";
const MODEL_LABEL = { rule: "Fixed formula", ridge: "Ridge", random_forest: "Random forest", gradient_boosting: "Gradient boosting" };
const BACKEND_NOTE = SBERT
  ? "Sentence-BERT (all-MiniLM-L6-v2) embeddings"
  : "Run with the lexical TF-IDF backend; re-run train.py with Sentence-BERT for final numbers";

const f2 = (x) => x.toFixed(2);
const pct = (x) => `${Math.round(x * 100)}%`;
const fmtMark = (x) => (Number.isInteger(x) ? `${x}` : x.toFixed(1));

// ------------------------------------------------------------------ design
const C = {
  navy: "1B2440", navy2: "2C3A63", ink: "1F2937", muted: "5B6475", line: "D9DEE7",
  paper: "F3F5F9", white: "FFFFFF", red: "C8414F", green: "2E8B5E", amber: "D98E04",
  chart: "2F5DA8", chartSoft: "A9BDE0", chart2: "2BA383",
};
const HEAD = "Cambria", BODY = "Calibri";

async function icon(name, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color: `#${color}`, size }));
  const png = await sharp(Buffer.from(svg)).resize(size, size).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.title = "AI Answer Evaluation System";
  pres.author = "Anika";
  pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

  pres.defineSlideMaster({
    title: "Title (dark)",
    background: { color: C.navy },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 11.9, h: 1.9, fontFace: HEAD, fontSize: 48, bold: true, color: C.white, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.1, w: 11.0, h: 1.0, fontFace: BODY, fontSize: 18, color: "C9D2E6", valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  const footer = [
    { text: { text: "AI Answer Evaluation System", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontFace: BODY, fontSize: 9, color: C.muted, margin: 0 } } },
  ];
  pres.defineSlideMaster({
    title: "Content",
    background: { color: C.white },
    objects: [
      ...footer,
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontFace: HEAD, fontSize: 32, bold: true, color: C.navy, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 9, color: C.muted, align: "right" },
  });
  pres.defineSlideMaster({
    title: "Closing (dark)",
    background: { color: C.navy },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontFace: HEAD, fontSize: 32, bold: true, color: C.white, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 9, color: "8C97B3", align: "right" },
  });

  const I = {};
  for (const [k, n, col] of [
    ["clock", "FaClock", C.white], ["scale", "FaBalanceScale", C.white], ["comment", "FaCommentDots", C.white],
    ["check", "FaCheck", C.white], ["times", "FaTimes", C.white], ["brain", "FaBrain", C.white],
    ["camera", "FaCamera", C.white], ["keyboard", "FaKeyboard", C.white], ["list", "FaListUl", C.white],
    ["chart", "FaChartLine", C.white], ["db", "FaDatabase", C.white], ["layer", "FaLayerGroup", C.white],
    ["bulb", "FaLightbulb", C.white], ["warn", "FaExclamationTriangle", C.white], ["code", "FaCode", C.white],
    ["pen", "FaPenNib", C.white], ["file", "FaFileAlt", C.white], ["robot", "FaCogs", C.white],
  ]) I[k] = await icon(n, col);

  // helpers -----------------------------------------------------------
  const txt = (slide, text, o) => slide.addText(text, { isTextBox: true, fontFace: BODY, color: C.ink, fontSize: 14, margin: 0, valign: "top", ...o });
  const badge = (slide, img, x, y, d, fill, name) => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name + " circle" });
    const p = d * 0.24;
    slide.addImage({ data: img, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name + " icon" });
  };
  const card = (slide, x, y, w, h, fill = C.paper, name = "card") =>
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
  const arrow = (slide, x1, y1, x2, y2, color = C.muted) =>
    slide.addShape(pres.shapes.LINE, {
      x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001),
      flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 1.5, endArrowType: "triangle" },
    });
  const content = (title, section) => {
    const s = pres.addSlide({ masterName: "Content", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  const tick = (slide, x, y, ok, d = 0.32) => badge(slide, ok ? I.check : I.times, x, y, d, ok ? C.green : C.red, ok ? "tick" : "cross");

  // ================================================================ 1. Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "Title (dark)", sectionTitle: "Introduction" });
  s.addText("AI Answer Evaluation System", { placeholder: "title" });
  s.addText("Deep-learning-based marking and feedback for descriptive answers", { placeholder: "body" });
  txt(s, "Anika  ·  B.Tech CSE (Data Science)  ·  Manipal University Jaipur", { x: 0.7, y: 6.3, w: 11.0, h: 0.4, fontSize: 14, color: "8C97B3" });
  s.addNotes("Introduce the project: a system that marks descriptive answers by comparing them with a teacher's reference answer and rubric, and explains the mark with a concept checklist and feedback.");

  // ================================================================ 2. Problem
  s = content("Marking descriptive answers by hand is slow and uneven", "Introduction");
  const probs = [
    [I.clock, "Time", "Every answer has to be read in full, so large classes take days to mark and feedback arrives late."],
    [I.scale, "Consistency", "Different examiners, or one examiner late in a long session, can give different marks to the same answer."],
    [I.comment, "Feedback", "Students usually see a number, not which key points were missing from their answer."],
  ];
  probs.forEach(([img, h, b], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.6, 3.85, 2.75, C.paper, `${h} card`);
    badge(s, img, x + 0.35, 1.9, 0.7, C.navy, h);
    txt(s, h, { x: x + 0.35, y: 2.75, w: 3.2, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: C.navy });
    txt(s, b, { x: x + 0.35, y: 3.25, w: 3.2, h: 1.0, fontSize: 14, color: C.ink });
  });
  card(s, 0.6, 4.75, 12.1, 1.85, C.navy, "problem statement");
  txt(s, "PROBLEM STATEMENT", { x: 1.0, y: 4.98, w: 6, h: 0.3, fontSize: 11, bold: true, color: "8C97B3", charSpacing: 2 });
  txt(s, "Manual evaluation of descriptive answers is time-consuming and may introduce inconsistency. This project proposes a deep-learning-based system that uses transformer-based semantic understanding and concept coverage to estimate marks and give constructive feedback.",
    { x: 1.0, y: 5.32, w: 11.3, h: 1.15, fontSize: 16, color: C.white, italic: true });
  s.addNotes("The goal is to assist teachers, not replace them: the system gives a consistent first mark and an explanation the teacher can check.");

  // ================================================================ 3. Project pipeline
  s = content("Project pipeline: from marked answers to a working app", "Introduction");
  const pipe = [
    [I.db, "Collect data", "Questions, reference answers, rubrics and teacher-marked answers", "data/"],
    [I.layer, "Preprocess", "Normalise text, split sentences, stem words", "text_utils.py"],
    [I.brain, "Embed", "Sentence-BERT encodes answers, sentences and concepts", "embedder.py"],
    [I.list, "Extract features", "Similarity, concept coverage, completeness, length", "core.py"],
    [I.robot, "Train scorer", "Regressors learn from teacher marks, one question held out", "train.py"],
    [I.chart, "Evaluate", "MAE, RMSE, correlation, QWK and concept F1", "evaluate.py"],
    [I.code, "Deploy", "Streamlit app for typed, batch and handwritten answers", "app.py"],
  ];
  const PW = 1.56, PG = 0.195, PY = 1.6, PH = 3.35;
  pipe.forEach(([img, h, b, f], i) => {
    const x = 0.6 + i * (PW + PG);
    const hero = i === 2 || i === 4;
    card(s, x, PY, PW, PH, hero ? C.navy : C.paper, h + " stage");
    badge(s, img, x + (PW - 0.62) / 2, PY + 0.25, 0.62, hero ? C.red : C.navy, h);
    txt(s, `${i + 1}`, { x: x + 0.12, y: PY + 0.1, w: 0.4, h: 0.3, fontSize: 11, bold: true, color: hero ? "8C97B3" : C.muted });
    txt(s, h, { x: x + 0.1, y: PY + 1.0, w: PW - 0.2, h: 0.5, fontFace: HEAD, fontSize: 15, bold: true, color: hero ? C.white : C.navy, align: "center", valign: "middle" });
    txt(s, b, { x: x + 0.12, y: PY + 1.55, w: PW - 0.24, h: 1.25, fontSize: 12, color: hero ? "D7DDEA" : C.ink, align: "center" });
    txt(s, f, { x: x + 0.05, y: PY + 2.9, w: PW - 0.1, h: 0.3, fontSize: 10, fontFace: "Courier New", color: hero ? "C9D2E6" : C.muted, align: "center" });
    if (i < pipe.length - 1) arrow(s, x + PW + 0.02, PY + PH / 2, x + PW + PG - 0.02, PY + PH / 2, C.navy);
  });
  const phases = [["Data preparation", 0, 1], ["Modelling", 2, 4], ["Validation and deployment", 5, 6]];
  phases.forEach(([label, a, b]) => {
    const x = 0.6 + a * (PW + PG), w = (b - a + 1) * PW + (b - a) * PG;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.15, w, h: 0.42, rectRadius: 0.06, fill: { color: "E6EBF4" }, line: { color: "E6EBF4" }, objectName: label + " phase" });
    txt(s, label, { x, y: 5.15, w, h: 0.42, fontSize: 13, bold: true, color: C.navy, align: "center", valign: "middle" });
  });
  card(s, 0.6, 5.85, 12.1, 0.8, C.paper, "pipeline note");
  txt(s, [{ text: "Iterative loop: ", options: { bold: true, color: C.navy } },
          { text: "as more teacher-marked answers are collected, steps 1 and 4–6 are re-run, and the app picks up the newly trained scorer automatically.", options: { color: C.ink } }],
    { x: 0.9, y: 5.85, w: 11.5, h: 0.8, fontSize: 14, valign: "middle" });
  s.addNotes("End-to-end workflow of the project. Each stage maps to one file in the code. Stages 3 and 5 are the machine-learning core: the pretrained transformer and the scorer trained on teacher marks.");

  // ================================================================ 4. Architecture
  pres.addSection({ title: "Method" });
  s = content("Architecture: three signals feed one scoring engine", "Method");
  const box = (x, y, w, h, title, sub, fill = C.paper, color = C.navy, name = title) => {
    card(s, x, y, w, h, fill, name);
    txt(s, [{ text: title, options: { bold: true, fontSize: 14, color, breakLine: !!sub } },
            ...(sub ? [{ text: sub, options: { fontSize: 11, color: fill === C.paper ? C.muted : "C9D2E6" } }] : [])],
      { x: x + 0.15, y, w: w - 0.3, h, valign: "middle", align: "center" });
  };
  box(0.6, 1.75, 2.1, 0.85, "Typed answer", "");
  box(0.6, 2.85, 2.1, 1.0, "Handwritten photo", "→ TrOCR transcription");
  box(0.6, 4.1, 2.1, 1.0, "Reference answer", "+ rubric concepts");
  box(3.15, 2.85, 2.0, 1.0, "Preprocessing", "normalise, split, stem");
  box(5.6, 2.6, 2.1, 1.5, "Sentence-BERT", "transformer encoder", C.navy, C.white);
  box(8.15, 1.6, 2.3, 0.95, "Semantic similarity", "cosine to reference");
  box(8.15, 2.87, 2.3, 0.95, "Concept coverage", "weighted rubric hits");
  box(8.15, 4.14, 2.3, 0.95, "Completeness", "reference sentences matched");
  box(10.9, 2.2, 1.85, 1.0, "Scoring engine", "learned regressor", C.navy, C.white);
  box(10.9, 3.75, 1.85, 1.35, "Marks", "+ checklist + feedback", C.red, C.white);
  arrow(s, 2.7, 2.17, 3.15, 3.2); arrow(s, 2.7, 3.35, 3.15, 3.35); arrow(s, 2.7, 4.6, 3.15, 3.5);
  arrow(s, 5.15, 3.35, 5.6, 3.35);
  arrow(s, 7.7, 3.1, 8.15, 2.07); arrow(s, 7.7, 3.35, 8.15, 3.35); arrow(s, 7.7, 3.6, 8.15, 4.6);
  arrow(s, 10.45, 2.07, 10.9, 2.55); arrow(s, 10.45, 3.35, 10.9, 2.75); arrow(s, 10.45, 4.6, 10.9, 2.95);
  arrow(s, 11.82, 3.2, 11.82, 3.75);
  card(s, 0.6, 5.55, 12.15, 1.1, C.paper, "architecture note");
  badge(s, I.brain, 0.85, 5.75, 0.7, C.navy, "deep learning");
  txt(s, "Deep learning sits in the two pretrained transformers: Sentence-BERT turns text into meaning vectors, and TrOCR reads handwriting. The scoring engine is a smaller model trained on teacher marks to weigh the three signals.",
    { x: 1.8, y: 5.7, w: 10.7, h: 0.85, fontSize: 14, valign: "middle" });
  s.addNotes("Walk left to right. Inputs are normalised and split into sentences, encoded by Sentence-BERT, turned into three signals, and combined by the scoring engine into marks with a checklist and feedback.");

  // ================================================================ 5. Similarity
  s = content("Stage 1: Sentence-BERT measures meaning, not word overlap", "Method");
  txt(s, [
    { text: "Each answer becomes a 384-number vector (an embedding) from the pretrained all-MiniLM-L6-v2 model.", options: { bullet: true, breakLine: true } },
    { text: "Answers that mean the same thing point in the same direction, even with different wording.", options: { bullet: true, breakLine: true } },
    { text: "Cosine similarity compares the directions; it is rescaled to 0–100% so scores from different models line up.", options: { bullet: true, breakLine: true } },
    { text: "The same embeddings are reused for concept matching and completeness.", options: { bullet: true } },
  ], { x: 0.6, y: 1.65, w: 5.6, h: 3.5, fontSize: 16, paraSpaceAfter: 10 });
  const vec = (yy, label, v) => {
    card(s, 6.9, yy, 5.8, 0.95, C.paper, label + " vector");
    txt(s, label, { x: 7.15, y: yy, w: 2.0, h: 0.95, fontSize: 14, bold: true, color: C.navy, valign: "middle" });
    txt(s, v, { x: 9.1, y: yy, w: 3.4, h: 0.95, fontSize: 15, fontFace: "Courier New", color: C.ink, valign: "middle" });
  };
  vec(1.7, "Reference", "[0.19, −0.40, 0.75, …]");
  vec(2.85, "Student", "[0.21, −0.43, 0.78, …]");
  card(s, 6.9, 4.1, 5.8, 1.25, C.navy, "cosine formula");
  txt(s, [{ text: "cos θ = (r · s) / (‖r‖ ‖s‖)", options: { fontSize: 20, color: C.white, fontFace: HEAD, breakLine: true } },
          { text: "≈ 0.9 similar meaning   ·   ≈ 0.3 different topic", options: { fontSize: 13, color: "C9D2E6" } }],
    { x: 7.15, y: 4.1, w: 5.4, h: 1.25, valign: "middle" });
  card(s, 0.6, 5.65, 12.1, 1.0, "FBEAEC", "similarity caveat");
  badge(s, I.warn, 0.85, 5.8, 0.7, C.red, "caveat");
  txt(s, "Similarity alone is not enough: a short answer with the right gist can score high while missing most of the points the teacher expects.",
    { x: 1.8, y: 5.65, w: 10.7, h: 1.0, fontSize: 14, valign: "middle" });
  s.addNotes("Vectors are illustrative. In code: evaluator/embedder.py. The calibration floor and ceiling per model are in PROFILES.");

  // ================================================================ 6. Concepts
  s = content("Stage 2: a rubric checks which key points are present", "Method");
  const q1 = questions[0];
  txt(s, "Rubric for “" + q1.question + "”", { x: 0.6, y: 1.55, w: 6, h: 0.35, fontSize: 14, bold: true, color: C.navy });
  q1.concepts.forEach((c, i) => {
    const yy = 2.0 + i * 0.62;
    card(s, 0.6, yy, 5.9, 0.5, C.paper, c.id + " row");
    txt(s, c.name.replace(" each other's", ""), { x: 0.8, y: yy, w: 4.4, h: 0.5, fontSize: 13, valign: "middle" });
    txt(s, `weight ${c.weight}`, { x: 5.2, y: yy, w: 1.15, h: 0.5, fontSize: 12, color: c.weight > 1 ? C.red : C.muted, bold: c.weight > 1, align: "right", valign: "middle" });
  });
  const steps = [
    [I.list, "Keyword match", "Text is lower-cased and stemmed, so “connection-oriented” matches “connection oriented” and “inserting” matches “insertion”."],
    [I.brain, "Semantic match", "If no keyword appears, each student sentence is compared with the concept using Sentence-BERT (cosine ≥ 0.55), which catches paraphrases."],
    [I.chart, "Weighted coverage", "Coverage = weight of concepts found ÷ total weight. Core ideas can carry more weight than details."],
  ];
  steps.forEach(([img, h, b], i) => {
    const yy = 1.6 + i * 1.7;
    badge(s, img, 7.0, yy, 0.7, C.navy, h);
    txt(s, h, { x: 7.9, y: yy, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.navy });
    txt(s, b, { x: 7.9, y: yy + 0.45, w: 4.8, h: 1.1, fontSize: 14 });
  });
  s.addNotes("Concepts come from the teacher. Detection code is in evaluator/core.py (extract_features). Each detected concept records how it was found and the evidence, which the app shows.");

  // ================================================================ 7. Two answers, same mark
  const e1 = examples.A04, e2 = examples.A05;
  s = content("Stage 3: the signals expose different weaknesses", "Method");
  txt(s, "Two answers to the deadlock question get the same mark for opposite reasons. Their similarity is almost equal; concept coverage shows which points each one has, and completeness (reference sentences matched, blended with length) shows both are too thin.",
    { x: 0.6, y: 1.5, w: 12.1, h: 0.8, fontSize: 15, color: C.muted });
  [[e1, "Explains the idea, misses the conditions"], [e2, "Lists the conditions, explains nothing"]].forEach(([e, h], i) => {
    const x = 0.6 + i * 6.2;
    card(s, x, 2.5, 5.9, 4.15, C.paper, h);
    txt(s, h, { x: x + 0.3, y: 2.7, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: C.navy });
    txt(s, "“" + e.answer + "”", { x: x + 0.3, y: 3.15, w: 5.3, h: 1.05, fontSize: 13, italic: true });
    [["Similarity", e.semantic_similarity], ["Concept coverage", e.concept_coverage], ["Completeness", e.completeness]].forEach(([k, v], j) => {
      const yy = 4.35 + j * 0.45;
      txt(s, k, { x: x + 0.3, y: yy, w: 1.9, h: 0.3, fontSize: 13, color: C.muted, valign: "middle" });
      s.addShape(pres.shapes.RECTANGLE, { x: x + 2.25, y: yy + 0.07, w: 2.6, h: 0.16, fill: { color: C.line }, line: { color: C.line } });
      s.addShape(pres.shapes.RECTANGLE, { x: x + 2.25, y: yy + 0.07, w: Math.max(0.02, 2.6 * v), h: 0.16, fill: { color: C.chart }, line: { color: C.chart } });
      txt(s, pct(v), { x: x + 4.95, y: yy, w: 0.65, h: 0.3, fontSize: 13, bold: true, align: "right", valign: "middle" });
    });
    txt(s, [{ text: `System ${fmtMark(e.cv_score)}`, options: { bold: true, color: C.navy } },
            { text: `   ·   Teacher ${fmtMark(e.teacher_score)}`, options: { color: C.muted } }],
      { x: x + 0.3, y: 5.85, w: 5.3, h: 0.5, fontSize: 16, valign: "middle" });
  });
  s.addNotes("Left answer: right gist, one of six concepts. Right answer: five of six concepts named but no explanation, so completeness is low. The feedback module also flags answers that list terms without explaining them.");

  // ================================================================ 8. Scoring engine
  s = content("The scoring engine learns how teachers weigh the signals", "Method");
  card(s, 0.6, 1.6, 5.9, 2.25, C.paper, "baseline formula");
  txt(s, "Baseline: fixed formula", { x: 0.9, y: 1.8, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.navy });
  txt(s, "Score = 10 × (0.5·similarity + 0.3·coverage + 0.2·completeness)", { x: 0.9, y: 2.35, w: 5.3, h: 0.6, fontSize: 14, fontFace: "Courier New" });
  txt(s, "Transparent, but the weights are guesses.", { x: 0.9, y: 3.1, w: 5.3, h: 0.5, fontSize: 14, color: C.muted });
  card(s, 0.6, 4.05, 5.9, 2.6, C.navy, "learned scorer");
  txt(s, "Learned scorer", { x: 0.9, y: 4.25, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.white });
  txt(s, [
    { text: "6 features: similarity, raw cosine, coverage, keyword-only share, completeness, length ratio", options: { bullet: true, breakLine: true } },
    { text: "Ridge, random forest and gradient boosting compared; the lowest-error model is saved", options: { bullet: true, breakLine: true } },
    { text: `Selected: ${MODEL_LABEL[best].toLowerCase()}`, options: { bullet: true } },
  ], { x: 0.9, y: 4.75, w: 5.35, h: 1.8, fontSize: 14, color: C.white, paraSpaceAfter: 6 });
  txt(s, "Validation: leave one question out", { x: 7.0, y: 1.6, w: 5.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.navy });
  txt(s, "The model is trained on five questions and tested on the sixth, then the test question rotates. It is always marked on a question it has never seen, as it would be in a real exam.",
    { x: 7.0, y: 2.1, w: 5.7, h: 1.2, fontSize: 14 });
  for (let r = 0; r < 6; r++) {
    txt(s, `Fold ${r + 1}`, { x: 7.0, y: 3.55 + r * 0.45, w: 0.9, h: 0.35, fontSize: 11, color: C.muted, valign: "middle" });
    for (let c = 0; c < 6; c++) {
      const test = r === c;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.95 + c * 0.78, y: 3.55 + r * 0.45, w: 0.7, h: 0.35, rectRadius: 0.05,
        fill: { color: test ? C.red : C.chartSoft }, line: { color: test ? C.red : C.chartSoft } });
      txt(s, `Q${c + 1}`, { x: 7.95 + c * 0.78, y: 3.55 + r * 0.45, w: 0.7, h: 0.35, fontSize: 10, color: test ? C.white : C.navy, align: "center", valign: "middle", bold: test });
    }
  }
  txt(s, [{ text: "■ ", options: { color: C.red } }, { text: "test question   ", options: { color: C.muted } },
          { text: "■ ", options: { color: C.chartSoft } }, { text: "training questions", options: { color: C.muted } }],
    { x: 7.95, y: 6.32, w: 4.7, h: 0.3, fontSize: 11 });
  s.addNotes("Code: train.py. Leave-one-question-out prevents the model from memorising the specific wording of a question's answers.");

  // ================================================================ 9. Handwriting
  s = content("Handwritten answers go through TrOCR first", "Method");
  const hw = [
    [I.camera, "Photo", "Student uploads a photo or scan of the answer sheet"],
    [I.layer, "Clean-up", "Greyscale, fix rotation, stretch contrast"],
    [I.list, "Line split", "Rows of ink are found with a projection profile"],
    [I.brain, "TrOCR", "Transformer reads each line (trocr-base-handwritten)"],
    [I.pen, "Review", "Teacher can correct the text before marking"],
  ];
  hw.forEach(([img, h, b], i) => {
    const x = 0.6 + i * 2.48;
    card(s, x, 1.7, 2.2, 3.0, C.paper, h + " step");
    badge(s, img, x + 0.75, 1.95, 0.7, i === 3 ? C.red : C.navy, h);
    txt(s, `${i + 1}. ${h}`, { x: x + 0.15, y: 2.85, w: 1.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: C.navy, align: "center" });
    txt(s, b, { x: x + 0.2, y: 3.3, w: 1.8, h: 1.3, fontSize: 13, align: "center" });
    if (i < 4) arrow(s, x + 2.22, 3.2, x + 2.46, 3.2, C.navy);
  });
  card(s, 0.6, 5.05, 12.1, 1.6, C.navy, "ocr note");
  txt(s, [
    { text: "Why typed answers first: ", options: { bold: true, color: C.white } },
    { text: "OCR errors flow straight into the score, so the evaluator was built and tested on typed text, and handwriting was added as a front end. TrOCR is pretrained on single lines, which is why the page is split into lines before recognition. Recognition accuracy on real answer sheets is still to be measured.", options: { color: "D7DDEA" } },
  ], { x: 0.95, y: 5.05, w: 11.4, h: 1.6, fontSize: 14, valign: "middle" });
  s.addNotes("Code: evaluator/ocr.py. The OCR model downloads on first use (about 1.3 GB). Future work: measure character error rate on a set of real answer sheets.");

  // ================================================================ 12. Scatter + error analysis
  pres.addSection({ title: "Results" });
  s = content("Errors come from meaning the model cannot read", "Results");
  const xs = cv.map((r) => +r.human_score), ys = cv.map((r) => +r[`${best}_score`]);
  s.addChart(pres.charts.SCATTER, [{ name: "Teacher", values: xs }, { name: "System", values: ys }], {
    x: 0.5, y: 1.45, w: 6.0, h: 5.3, lineSize: 0, lineDataSymbol: "circle", lineDataSymbolSize: 8,
    chartColors: [C.chart], showLegend: false,
    showTitle: true, title: "System mark vs teacher mark (each dot is one answer)", titleFontSize: 13, titleColor: C.ink, titleFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 10, valAxisMajorUnit: 2, catAxisMinVal: 0, catAxisMaxVal: 10, catAxisMajorUnit: 2,
    showValAxisTitle: true, valAxisTitle: "System mark", showCatAxisTitle: true, catAxisTitle: "Teacher mark",
    valAxisTitleFontSize: 12, catAxisTitleFontSize: 12, valAxisTitleColor: C.muted, catAxisTitleColor: C.muted,
    catAxisLabelColor: C.muted, valAxisLabelColor: C.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
    valGridLine: { color: "E5E8EE", size: 0.5 }, catGridLine: { color: "E5E8EE", size: 0.5 },
  });
  txt(s, "Largest disagreements", { x: 7.0, y: 1.55, w: 5.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.navy });
  const worst = cv.map((r) => ({ id: r.answer_id, h: +r.human_score, p: +r[`${best}_score`] }))
    .sort((a, b) => Math.abs(b.p - b.h) - Math.abs(a.p - a.h)).slice(0, 3);
  worst.forEach((w, i) => {
    const yy = 2.05 + i * 1.12;
    card(s, 7.0, yy, 5.7, 1.0, C.paper, w.id + " error");
    let t = answers[w.id].student_answer;
    if (t.length > 92) t = t.slice(0, 89).replace(/\s+\S*$/, "") + "…";
    txt(s, "“" + t + "”", { x: 7.2, y: yy + 0.08, w: 3.9, h: 0.85, fontSize: 12, italic: true, valign: "middle" });
    txt(s, [{ text: `System ${fmtMark(w.p)}`, options: { bold: true, color: w.p > w.h ? C.red : C.navy, breakLine: true } },
            { text: `Teacher ${fmtMark(w.h)}`, options: { color: C.muted } }],
      { x: 11.2, y: yy + 0.08, w: 1.35, h: 0.85, fontSize: 13, valign: "middle", align: "right" });
  });
  txt(s, SBERT
    ? "Read these answers to see what the embeddings still miss, then add rubric keywords or more training examples."
    : "Word-overlap features over-mark wrong answers that reuse the right vocabulary and under-mark correct paraphrases. This is the gap Sentence-BERT embeddings are meant to close.",
    { x: 7.0, y: 5.5, w: 5.7, h: 1.2, fontSize: 14, color: C.ink });
  s.addNotes("Dots on the diagonal are perfect agreement. The listed answers are the three with the largest cross-validated error.");

  // ================================================================ 13. Concept detection
  s = content(`Concept detection: ${pct(CD.recall)} recall, ${pct(CD.precision)} precision`, "Results");
  const qs = Object.keys(metrics.concept_detection_by_question).sort();
  const byQ = metrics.concept_detection_by_question;
  s.addChart(pres.charts.BAR, [
    { name: "Precision", labels: qs, values: qs.map((q) => +byQ[q].precision.toFixed(2)) },
    { name: "Recall", labels: qs, values: qs.map((q) => +byQ[q].recall.toFixed(2)) },
  ], {
    x: 0.5, y: 1.5, w: 7.2, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [C.chart, C.chart2],
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: C.ink, legendFontFace: "+mn-lt",
    showTitle: true, title: "Concept detection by question", titleFontSize: 13, titleColor: C.ink, titleFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", valAxisMajorUnit: 0.25,
    catAxisLabelColor: C.muted, valAxisLabelColor: C.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E5E8EE", size: 0.5 }, catGridLine: { style: "none" }, barGapWidthPct: 70,
  });
  [[f2(CD.f1), "F1 score across all answers"], [`${CD.fn}`, SBERT ? "key points missed" : "concepts missed, mostly paraphrases"], [`${CD.fp}`, "false detections"]].forEach(([v, l], i) => {
    const yy = 1.7 + i * 1.05;
    txt(s, v, { x: 8.2, y: yy, w: 1.6, h: 0.8, fontFace: HEAD, fontSize: 40, bold: true, color: C.navy, valign: "middle" });
    txt(s, l, { x: 9.85, y: yy, w: 2.85, h: 0.8, fontSize: 14, color: C.muted, valign: "middle" });
  });
  txt(s, SBERT
    ? "Sentence-BERT matching catches paraphrases that contain no rubric keyword, which raises recall, but it also flags some points that are not really there, which costs precision. Keywords were written by the same author who labelled the answers, so these figures are likely optimistic."
    : "Example miss: “resending lost packets” means reliable delivery but contains no rubric keyword. Semantic matching with Sentence-BERT targets exactly these cases. Keywords were written by the same author who labelled the answers, so these figures are likely optimistic.",
    { x: 8.2, y: 4.95, w: 4.5, h: 1.6, fontSize: 13 });
  s.addNotes("Precision, recall and F1 compare detected concepts with the gold concept labels in student_answers.csv.");

  // ================================================================ 15. Tech stack & structure
  pres.addSection({ title: "Application" });
  s = content("Built with open-source Python tools", "Application");
  const stack = [["Sentence-Transformers", "semantic embeddings"], ["PyTorch", "model runtime"], ["Hugging Face TrOCR", "handwriting recognition"],
                 ["scikit-learn", "learned scorer, metrics"], ["NLTK", "Porter stemming"], ["Streamlit", "web application"],
                 ["pandas · NumPy · SciPy", "data and statistics"], ["matplotlib", "report charts"]];
  stack.forEach(([h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 3.05, yy = 1.6 + row * 1.25;
    card(s, x, yy, 2.85, 1.05, C.paper, h);
    txt(s, h, { x: x + 0.2, y: yy + 0.15, w: 2.5, h: 0.4, fontSize: 15, bold: true, color: C.navy });
    txt(s, b, { x: x + 0.2, y: yy + 0.55, w: 2.5, h: 0.35, fontSize: 12, color: C.muted });
  });
  card(s, 6.95, 1.6, 5.75, 4.85, C.navy, "project tree");
  txt(s, [
    "answer-evaluator/", "├─ evaluator/", "│  ├─ core.py        evaluator", "│  ├─ embedder.py    Sentence-BERT", "│  ├─ scorer.py      scoring engines",
    "│  ├─ feedback.py    feedback text", "│  ├─ ocr.py         TrOCR", "│  └─ metrics.py     MAE, QWK, F1", "├─ data/            questions, answers",
    "├─ train.py         train scorer", "├─ evaluate.py      metrics, charts", "├─ finetune_sbert.py", "├─ app.py           Streamlit app", "└─ tests/",
  ].map((l, i, a) => ({ text: l, options: { breakLine: i < a.length - 1 } })),
    { x: 7.25, y: 1.8, w: 5.3, h: 4.5, fontFace: "Courier New", fontSize: 12, color: "E3E8F3" });
  s.addNotes("All code is in the project folder; README.md has setup steps.");

  // ================================================================ 17. Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "Closing (dark)", sectionTitle: "Conclusion" });
  s.addText("Conclusion", { placeholder: "title" });
  const takeaways = [
    [I.brain, "Meaning, not keywords", "Transformer embeddings compare what an answer says, and a rubric checks which key points it covers."],
    [I.robot, "Learns from teachers", `A scorer trained on marked answers cut the error from ${f2(metrics.scoring.rule.mae)} to ${f2(S.mae)} marks on unseen questions.`],
    [I.comment, "Explains every mark", "Each score comes with a concept checklist, evidence and feedback a student can act on."],
  ];
  takeaways.forEach(([img, h, b], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.7, 3.85, 3.3, C.navy2, h);
    badge(s, img, x + 0.35, 2.0, 0.75, C.red, h);
    txt(s, h, { x: x + 0.35, y: 2.95, w: 3.2, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: C.white });
    txt(s, b, { x: x + 0.35, y: 3.5, w: 3.2, h: 1.4, fontSize: 14, color: "D7DDEA" });
  });
  txt(s, "Thank you", { x: 0.6, y: 5.6, w: 12.1, h: 0.8, fontFace: HEAD, fontSize: 30, bold: true, color: C.white, align: "center", valign: "middle" });
  s.addNotes("Close with the three takeaways and invite questions. Offer the live demo if time allows.");

  await pres.writeFile({ fileName: OUT });
  console.log("Wrote " + OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
