import express from 'express';
import cors from 'cors';
import multer from 'multer';
import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';

const app = express();
const PORT = Number(process.env.PORT || 8000);
const ML_URL = process.env.ML_URL || 'http://127.0.0.1:8001';
const PROJECT_ROOT = path.resolve(process.cwd(), '..');
const DATA_DIR = path.resolve('data');
fs.mkdirSync(DATA_DIR, { recursive: true });
const db = new Database(path.join(DATA_DIR, 'pulseai.db'));
db.pragma('journal_mode = WAL');
db.exec(`CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file TEXT NOT NULL,
  created_at TEXT NOT NULL,
  beats INTEGER,
  status TEXT NOT NULL,
  result_json TEXT
)`);
db.exec(`CREATE TABLE IF NOT EXISTS training_records (
  record_id TEXT PRIMARY KEY,
  split TEXT NOT NULL,
  dataset TEXT NOT NULL,
  status TEXT NOT NULL
)`);
const trainingRecords = [
  ['100', 'train'], ['101', 'train'], ['103', 'train'], ['105', 'train'],
  ['106', 'train'], ['108', 'train'], ['109', 'train'], ['111', 'train'],
  ['112', 'train'], ['113', 'train'], ['115', 'train'], ['116', 'train'],
  ['102', 'test'], ['104', 'test'], ['107', 'test'], ['117', 'test'],
  ['118', 'test'], ['119', 'test']
];
const seedTrainingRecord = db.prepare("INSERT OR IGNORE INTO training_records (record_id, split, dataset, status) VALUES (?, ?, 'MIT-BIH Arrhythmia Database', 'configured')");
const seedTrainingRecords = db.transaction(() => trainingRecords.forEach(([recordId, split]) => seedTrainingRecord.run(recordId, split)));
seedTrainingRecords();
const insertAnalysis = db.prepare('INSERT INTO analyses (file, created_at, beats, status, result_json) VALUES (?, ?, ?, ?, ?)');
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });
const DOCUMENT_EXTENSIONS = new Set(['.pdf', '.png', '.jpg', '.jpeg', '.webp']);
app.use(cors({ origin: ['http://localhost:5173', 'http://127.0.0.1:5173'] }));
app.use(express.json({ limit: '1mb' }));

app.get('/api/health', async (_req, res) => {
  let ml = { available: false, model_loaded: false };
  try { const r = await fetch(`${ML_URL}/health`, { signal: AbortSignal.timeout(2500) }); if (r.ok) ml = await r.json(); } catch {}
  res.json({ status: 'ok', service: 'pulseai-api', ml_service_available: Boolean(ml.available ?? true), model_loaded: Boolean(ml.model_loaded), timestamp: new Date().toISOString() });
});
app.get('/api/history', (_req, res) => {
  const history = db.prepare('SELECT id, file, created_at, beats, status, result_json FROM analyses ORDER BY id DESC LIMIT 100')
    .all()
    .map((r) => {
      let flagged = null;
      try {
        const predictions = JSON.parse(r.result_json || '{}').predictions;
        if (Array.isArray(predictions)) flagged = predictions.filter((prediction) => prediction.class !== 'N').length;
      } catch (err) {
        console.error(`Could not read stored analysis ${r.id}:`, err);
      }
      return { id: `PA-${String(r.id).padStart(4, '0')}`, file: r.file, created_at: r.created_at, beats: r.beats, flagged, status: r.status };
    });
  res.json({ history });
});
app.get('/api/training-data', (_req, res) => {
  const records = db.prepare('SELECT record_id, split, dataset, status FROM training_records ORDER BY split DESC, record_id').all();
  const trainedArtifact = [
    path.join(PROJECT_ROOT, 'ml-service', 'models', 'ecg_cnn.keras'),
    path.join(PROJECT_ROOT, 'training', 'artifacts', 'ecg_cnn.keras')
  ].some((candidate) => fs.existsSync(candidate));
  res.json({
    dataset: 'MIT-BIH Arrhythmia Database',
    dataset_url: 'https://physionet.org/content/mitdb/1.0.0/',
    record_count: records.length,
    train_count: records.filter((record) => record.split === 'train').length,
    test_count: records.filter((record) => record.split === 'test').length,
    records,
    training_status: trainedArtifact ? 'artifact_available' : 'configured_not_trained',
    note: trainedArtifact
      ? 'A model artifact is present; see evaluation metrics for measured performance.'
      : 'These records are the configured small-project split. Run the training script to create a model artifact.'
  });
});
app.get('/api/metrics', (_req, res) => {
  const candidates = [
    path.join(PROJECT_ROOT, 'ml-service', 'models', 'metrics.json'),
    path.join(PROJECT_ROOT, 'training', 'artifacts', 'metrics.json')
  ];
  const metricsPath = candidates.find((candidate) => fs.existsSync(candidate));
  if (!metricsPath) return res.json({ metrics: null, source: 'unavailable' });
  try {
    res.json({
      metrics: JSON.parse(fs.readFileSync(metricsPath, 'utf8')),
      source: 'training-artifact',
      path: path.relative(PROJECT_ROOT, metricsPath)
    });
  } catch (err) {
    console.error(`Could not read metrics artifact ${metricsPath}:`, err);
    res.status(500).json({ error: 'metrics.json could not be read' });
  }
});
app.post('/api/analyze', upload.single('ecg'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'Upload a CSV, PDF, or ECG image in the ecg field.' });
  const extension = path.extname(req.file.originalname).toLowerCase();
  if (extension !== '.csv' && !DOCUMENT_EXTENSIONS.has(extension)) {
    return res.status(415).json({ error: 'Supported uploads are CSV, PDF, PNG, JPG, JPEG, and WEBP files.' });
  }
  if (extension !== '.csv') {
    const createdAt = new Date().toISOString();
    const result = {
      file_type: extension.slice(1).toUpperCase(),
      status: 'uploaded',
      message: 'File uploaded successfully. PDF and image files are stored as research attachments; ECG classification currently requires a numeric CSV signal.',
      filename: req.file.originalname,
      size_bytes: req.file.size
    };
    const info = insertAnalysis.run(req.file.originalname, createdAt, null, 'Uploaded (not analysed)', JSON.stringify(result));
    return res.status(202).json({
      analysis: {
        id: `PA-${String(info.lastInsertRowid).padStart(4, '0')}`,
        file: req.file.originalname,
        created_at: createdAt,
        beats: null,
        status: 'Uploaded (not analysed)'
      },
      ...result
    });
  }
  const text = req.file.buffer.toString('utf8');
  const values = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const first = trimmed.split(/[;,]/)[0].trim();
    const n = Number(first);
    if (Number.isFinite(n)) values.push(n);
  }
  if (values.length < 100) return res.status(400).json({ error: 'CSV must contain at least 100 numeric ECG samples in the first column.' });
  if (values.length > 2_000_000) return res.status(413).json({ error: 'Signal is too long for this prototype. Use at most 2,000,000 samples.' });
  try {
    const r = await fetch(`${ML_URL}/predict`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ signal: values }),
      signal: AbortSignal.timeout(120000)
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) return res.status(r.status).json({ error: data.detail || data.error || 'ML inference failed. Check the model and preprocessing.' });
    const result = data;
    const beats = Array.isArray(result.predictions) ? result.predictions.length : null;
    const info = insertAnalysis.run(req.file.originalname, new Date().toISOString(), beats, 'Completed', JSON.stringify(result));
    res.json({ analysis: { id: `PA-${String(info.lastInsertRowid).padStart(4, '0')}`, file: req.file.originalname, created_at: new Date().toISOString(), beats, status: 'Completed' }, ...result });
  } catch (err) {
    res.status(503).json({ error: `ML service unavailable at ${ML_URL}. Start the Python service and confirm a trained model is installed.` });
  }
});
app.get('/api', (_req, res) => res.json({ name: 'PulseAI API', version: '1.0.0', disclaimer: 'Research prototype only; not a medical diagnostic tool.' }));
app.listen(PORT, () => console.log(`PulseAI API listening on http://localhost:${PORT}`));
