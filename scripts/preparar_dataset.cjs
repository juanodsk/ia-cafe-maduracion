/* Reproducible 70/15/15 COCO preparation, using only Node.js built-in modules. */
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const ROOT = path.resolve(__dirname, '..');
const SPLITS = ['train', 'valid', 'test'];
const RATIOS = [0.70, 0.15, 0.15];
const ALGORITHM = 'grouped-multiclass-exact-image-counts-v1';
const SEED = 42;
const TRIALS = 256;
const SWAPS = 100000;
const CLASSES = ['dry', 'overripe', 'ripe', 'semi_ripe', 'unripe'];
const source = path.resolve(ROOT, process.argv[2] || 'imagenes');
const output = path.resolve(ROOT, process.argv[3] || 'dataset_preparado');
const compare = (a, b) => a < b ? -1 : a > b ? 1 : 0;
const sum = values => values.reduce((a, b) => a + b, 0);
const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
const portable = value => value.split(path.sep).join('/');
const csvCell = value => {
  const s = String(value ?? '');
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};
function csv(file, fields, rows) {
  fs.writeFileSync(file, [fields.join(','), ...rows.map(row => fields.map(f => csvCell(row[f])).join(','))].join('\n') + '\n');
}
function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffled(values, random) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
function captureKey(name) {
  const stem = name.split('.rf.')[0].replace(/_(?:jpg|jpeg|png)$/i, '');
  const match = stem.match(/(\d{8})[_-](\d{6})(?!\d)/);
  return match ? 'capture:' + match[1] + '_' + match[2] : 'source:' + stem;
}
function integerTargets(n) {
  const raw = RATIOS.map(r => n * r);
  const targets = raw.map(Math.floor);
  const order = [0, 1, 2].sort((a, b) => (raw[b] - targets[b]) - (raw[a] - targets[a]) || a - b);
  for (let remaining = n - sum(targets), i = 0; remaining > 0; i++, remaining--) targets[order[i]]++;
  return targets;
}
function exactSubset(groups, target, random) {
  const order = shuffled(groups, random);
  const previous = new Int32Array(target + 1).fill(-1);
  const chosen = new Int32Array(target + 1).fill(-1);
  previous[0] = 0;
  for (let i = 0; i < order.length; i++) {
    const size = order[i].records.length;
    for (let n = target; n >= size; n--) {
      if (previous[n] === -1 && previous[n - size] !== -1) {
        previous[n] = n - size;
        chosen[n] = i;
      }
    }
    if (previous[target] !== -1) break;
  }
  assert.notEqual(previous[target], -1, 'Cannot achieve exact requested image counts with these groups.');
  const result = new Set();
  for (let n = target; n > 0; n = previous[n]) result.add(order[chosen[n]].id);
  return result;
}
function classCounts(groups, assignment) {
  const counts = SPLITS.map(() => CLASSES.map(() => 0));
  for (const g of groups) for (let c = 0; c < CLASSES.length; c++) counts[assignment[g.id]][c] += g.counts[c];
  return counts;
}
function objective(counts, totals) {
  let result = 0;
  for (let s = 0; s < 3; s++) for (let c = 0; c < CLASSES.length; c++) {
    const deviation = counts[s][c] / totals[c] - RATIOS[s];
    result += deviation * deviation / RATIOS[s];
  }
  return result;
}

assert(output !== source && !output.startsWith(source + path.sep), 'Output must be separate from the original download.');
assert(!fs.existsSync(output), 'Output already exists. Choose a new output folder to preserve the saved partition.');
const records = [];
const sourceManifest = [];
let categories, licenses;
for (const split of SPLITS) {
  const annotationPath = path.join(source, split, '_annotations.coco.json');
  const content = fs.readFileSync(annotationPath);
  const data = JSON.parse(content);
  sourceManifest.push({path: split + '/_annotations.coco.json', sha256: sha256(content)});
  if (!categories) { categories = data.categories; licenses = data.licenses; }
  else assert.deepEqual(data.categories, categories, 'Categories differ across source partitions.');
  const images = new Map();
  const categoryNames = new Map(data.categories.map(c => [c.id, c.name]));
  for (const image of data.images) {
    assert(!images.has(image.id), 'Duplicate source COCO image ID.');
    assert.equal(path.basename(image.file_name), image.file_name, 'Expected flat COCO image names.');
    const imagePath = path.join(source, split, image.file_name);
    const bytes = fs.readFileSync(imagePath);
    assert.equal(bytes[0], 0xff, 'Missing JPEG start marker.');
    assert.equal(bytes[1], 0xd8, 'Missing JPEG start marker.');
    const record = {key: split + '/' + image.id, source_split: split, source_path: split + '/' + image.file_name,
      source_name: image.file_name.split('.rf.')[0], capture_key: captureKey(image.file_name), image,
      image_sha256: sha256(bytes), annotations: [], class_counts: CLASSES.map(() => 0)};
    images.set(image.id, record);
    records.push(record);
  }
  for (const annotation of data.annotations) {
    const record = images.get(annotation.image_id);
    assert(record, 'Annotation references an unknown source image.');
    const label = categoryNames.get(annotation.category_id);
    assert(CLASSES.includes(label), 'Unexpected annotated category: ' + label);
    assert(annotation.bbox.length === 4 && annotation.bbox.every(Number.isFinite) && annotation.bbox[2] > 0 && annotation.bbox[3] > 0);
    assert(Array.isArray(annotation.segmentation) && annotation.segmentation.length > 0, 'Expected polygon segmentations.');
    record.annotations.push(annotation);
    record.class_counts[CLASSES.indexOf(label)]++;
  }
}

// Unite capture aliases, Roboflow variants and exact file duplicates before deduplication.
const parent = new Map();
function find(key) {
  if (!parent.has(key)) parent.set(key, key);
  if (parent.get(key) !== key) parent.set(key, find(parent.get(key)));
  return parent.get(key);
}
function unite(a, b) {
  a = find(a); b = find(b);
  if (a !== b) { const first = compare(a, b) < 0 ? a : b; parent.set(a === first ? b : a, first); }
}
const byHash = new Map();
for (const record of records) {
  find(record.capture_key);
  if (!byHash.has(record.image_sha256)) byHash.set(record.image_sha256, []);
  byHash.get(record.image_sha256).push(record);
}
const retained = [];
const duplicates = [];
const preference = {valid: 0, test: 1, train: 2};
for (const equal of byHash.values()) {
  equal.sort((a, b) => preference[a.source_split] - preference[b.source_split] || compare(a.source_path, b.source_path));
  const kept = equal[0];
  retained.push(kept);
  for (const removed of equal.slice(1)) {
    unite(kept.capture_key, removed.capture_key);
    assert.deepEqual(kept.class_counts, removed.class_counts, 'Exact duplicate has conflicting class counts; review manually.');
    const normalizedAnnotations = r => JSON.stringify(r.annotations.map(a => JSON.stringify({category_id: a.category_id, bbox: a.bbox, segmentation: a.segmentation})).sort());
    duplicates.push({kept_source_path: kept.source_path, excluded_source_path: removed.source_path,
      sha256: kept.image_sha256, excluded_annotations: removed.annotations.length,
      annotation_geometry_identical: normalizedAnnotations(kept) === normalizedAnnotations(removed),
      selection_rule: 'prefer_original_valid_then_test_then_train; keep_one_complete_annotation_set'});
  }
}
retained.sort((a, b) => compare(a.source_path, b.source_path));
const groupMap = new Map();
for (const r of retained) {
  const key = find(r.capture_key);
  if (!groupMap.has(key)) groupMap.set(key, {key, records: [], counts: CLASSES.map(() => 0)});
  const g = groupMap.get(key);
  g.records.push(r);
  r.group_key = key;
  for (let c = 0; c < CLASSES.length; c++) g.counts[c] += r.class_counts[c];
}
const groups = [...groupMap.values()].sort((a, b) => compare(a.key, b.key));
groups.forEach((g, id) => { g.id = id; g.public_id = 'g_' + sha256(g.key).slice(0, 16); });
const targets = integerTargets(retained.length);
const totals = CLASSES.map((_, c) => sum(groups.map(g => g.counts[c])));
assert(totals.every(n => n > 0));
const random = randomGenerator(SEED);
let bestAssignment, bestCounts, bestScore = Infinity;
for (let trial = 0; trial < TRIALS; trial++) {
  const valid = exactSubset(groups, targets[1], random);
  const test = exactSubset(groups.filter(g => !valid.has(g.id)), targets[2], random);
  const assignment = groups.map(g => valid.has(g.id) ? 1 : test.has(g.id) ? 2 : 0);
  const counts = classCounts(groups, assignment);
  const score = objective(counts, totals);
  if (score < bestScore) { bestScore = score; bestAssignment = assignment; bestCounts = counts; }
}

// Equal-size group swaps preserve exact file counts while improving class balance.
const bins = new Map();
for (const g of groups) {
  const size = g.records.length;
  if (!bins.has(size)) bins.set(size, [[], [], []]);
  bins.get(size)[bestAssignment[g.id]].push(g.id);
}
for (let iteration = 0; iteration < SWAPS; iteration++) {
  const a = groups[Math.floor(random() * groups.length)];
  const s = bestAssignment[a.id];
  const t = (s + 1 + Math.floor(random() * 2)) % 3;
  const pools = bins.get(a.records.length);
  if (pools[t].length === 0) continue;
  const bIndex = Math.floor(random() * pools[t].length);
  const b = groups[pools[t][bIndex]];
  const next = bestCounts.map(row => [...row]);
  for (let c = 0; c < CLASSES.length; c++) { next[s][c] += b.counts[c] - a.counts[c]; next[t][c] += a.counts[c] - b.counts[c]; }
  const score = objective(next, totals);
  if (score >= bestScore - 1e-15) continue;
  const aIndex = pools[s].indexOf(a.id);
  pools[s][aIndex] = b.id;
  pools[t][bIndex] = a.id;
  bestAssignment[a.id] = t;
  bestAssignment[b.id] = s;
  bestCounts = next;
  bestScore = score;
}
for (let s = 0; s < 3; s++) {
  assert.equal(sum(groups.filter(g => bestAssignment[g.id] === s).map(g => g.records.length)), targets[s]);
  assert(bestCounts[s].every(n => n > 0), 'Every maturity class must occur in every partition.');
}

fs.mkdirSync(output);
const csvDir = path.join(output, 'csv');
fs.mkdirSync(csvDir);
const imageRows = [], annotationRows = [], groupRows = [], distributionRows = [], classRows = [];
for (const g of groups) {
  g.split = SPLITS[bestAssignment[g.id]];
  groupRows.push({group_id: g.public_id, capture_key: g.key, split: g.split, images: g.records.length,
    source_names: [...new Set(g.records.map(r => r.source_name))].sort(compare).join('|'),
    ...Object.fromEntries(CLASSES.map((c, i) => [c + '_annotations', g.counts[i]]))});
  for (const r of g.records) { r.split = g.split; r.group_id = g.public_id; }
}
for (let s = 0; s < 3; s++) {
  const split = SPLITS[s];
  const dir = path.join(output, split);
  fs.mkdirSync(dir);
  const splitRecords = retained.filter(r => r.split === split).sort((a, b) => compare(a.image.file_name, b.image.file_name));
  const images = [], annotations = [];
  const filenames = new Set();
  for (const r of splitRecords) {
    assert(!filenames.has(r.image.file_name), 'Output filename collision.');
    filenames.add(r.image.file_name);
    const imageId = images.length + 1;
    images.push({...r.image, id: imageId});
    fs.copyFileSync(path.join(source, r.source_path), path.join(dir, r.image.file_name), fs.constants.COPYFILE_EXCL);
    imageRows.push({split, image_id: imageId, file_name: r.image.file_name, relative_path: split + '/' + r.image.file_name,
      source_path: r.source_path, source_split: r.source_split, source_image_id: r.image.id, source_name: r.source_name,
      group_id: r.group_id, capture_key: r.capture_key, sha256: r.image_sha256, width: r.image.width, height: r.image.height,
      annotation_count: r.annotations.length, ...Object.fromEntries(CLASSES.map((c, i) => [c + '_annotations', r.class_counts[i]]))});
    for (const a of r.annotations) {
      const annotationId = annotations.length + 1;
      annotations.push({...a, id: annotationId, image_id: imageId});
      const label = categories.find(c => c.id === a.category_id).name;
      annotationRows.push({split, annotation_id: annotationId, image_id: imageId, relative_image_path: split + '/' + r.image.file_name,
        group_id: r.group_id, category_id: a.category_id, label,
        bbox_x: a.bbox[0], bbox_y: a.bbox[1], bbox_width: a.bbox[2], bbox_height: a.bbox[3],
        source_split: r.source_split, source_image_id: r.image.id, source_annotation_id: a.id});
    }
  }
  const coco = {info: {description: 'Coffee Fruit Maturity v5; grouped 70/15/15 project split',
    version: 'v5-project-split-1', source_version: '5', preparation_algorithm: ALGORITHM, seed: SEED},
    licenses, categories, images, annotations};
  fs.writeFileSync(path.join(dir, '_annotations.coco.json'), JSON.stringify(coco));
  distributionRows.push({split, images: images.length, image_percentage: (100 * images.length / retained.length).toFixed(6),
    target_percentage: 100 * RATIOS[s], groups: groups.filter(g => g.split === split).length, annotations: annotations.length});
  for (let c = 0; c < CLASSES.length; c++) classRows.push({split, label: CLASSES[c], annotations: bestCounts[s][c],
    percentage_of_all_annotations_for_class: (100 * bestCounts[s][c] / totals[c]).toFixed(6),
    images_containing_class: splitRecords.filter(r => r.class_counts[c] > 0).length});
  csv(path.join(csvDir, split + '.csv'), Object.keys(imageRows[0]), imageRows.filter(r => r.split === split));
}
csv(path.join(csvDir, 'imagenes.csv'), Object.keys(imageRows[0]), imageRows);
csv(path.join(csvDir, 'anotaciones.csv'), Object.keys(annotationRows[0]), annotationRows);
csv(path.join(csvDir, 'grupos.csv'), Object.keys(groupRows[0]), groupRows);
csv(path.join(csvDir, 'distribucion.csv'), Object.keys(distributionRows[0]), distributionRows);
csv(path.join(csvDir, 'distribucion_clases.csv'), Object.keys(classRows[0]), classRows);
csv(path.join(csvDir, 'duplicados_excluidos.csv'), ['kept_source_path', 'excluded_source_path', 'sha256', 'excluded_annotations', 'annotation_geometry_identical', 'selection_rule'], duplicates);
fs.copyFileSync(path.join(source, 'README.dataset.txt'), path.join(output, 'README.dataset.original.txt'));
fs.copyFileSync(path.join(source, 'README.roboflow.txt'), path.join(output, 'README.roboflow.original.txt'));
const manifestDigest = sha256(fs.readFileSync(path.join(csvDir, 'imagenes.csv')));
const report = {schema_version: 1, algorithm: ALGORITHM, seed: SEED, search_trials: TRIALS, equal_size_swap_attempts: SWAPS,
  source_directory: portable(path.relative(ROOT, source)), output_directory: portable(path.relative(ROOT, output)),
  source_version: '5', source_annotation_files: sourceManifest,
  requested_ratios: Object.fromEntries(SPLITS.map((s, i) => [s, RATIOS[i]])), rounding: 'largest_remainder_on_unique_image_files',
  original_images: records.length, retained_images: retained.length, excluded_duplicate_files: duplicates.length,
  excluded_duplicate_annotations: sum(duplicates.map(d => d.excluded_annotations)), groups: groups.length,
  retained_annotations: sum(totals), class_labels: CLASSES, split_counts: distributionRows, class_counts: classRows,
  group_rule: 'same source stem or same YYYYMMDD_HHMMSS timestamp; union exact SHA256; keep whole groups in one split',
  duplicate_policy: 'Prefer existing valid, then test, then train; retain one annotation set and audit excluded geometry.',
  class_balance_objective: bestScore, image_manifest_sha256: manifestDigest,
  limitations: [
    'Source v5 already includes shear/exposure variants. This repartition keeps inherited variants grouped; it does not recover augmentation-free originals.',
    'Do not apply additional random augmentation to validation or test. For an evaluation on original images only, re-export a dataset without augmentation.',
    'Grouping controls source-name, timestamp-alias and exact-hash overlap. No farm, plant or physical-fruit IDs are available to guarantee independence across captures.',
    '70/15/15 refers to unique exported image files, not independent captures or fruit crops. All future crops must inherit the parent image partition.'
  ]};
fs.writeFileSync(path.join(output, 'preparacion.json'), JSON.stringify(report, null, 2) + '\n');
fs.writeFileSync(path.join(output, 'README.md'), '# Dataset preparado para madurez de café\n\n' +
  'Partición reproducible 70 % / 15 % / 15 %, semilla 42. Los conteos enteros son ' + targets.join(' / ') + ' imágenes.\n\n' +
  'Cada carpeta train, valid y test contiene imágenes y _annotations.coco.json. Los CSV de csv/ fijan la asignación, las distribuciones, los grupos y la trazabilidad por objeto.\n\n' +
  'La descarga original se conserva en imagenes/. Se excluyó una copia exacta del conjunto preparado; las imágenes relacionadas permanecen en el mismo conjunto.\n\n' +
  'Los recortes para CNN y Transformers deben heredar el split de su imagen. No redividir objetos individualmente.\n\n' +
  'La versión v5 ya contiene aumentos. Esta organización conserva esas variantes dentro de sus grupos y no identifica originales sin aumentos para validación/prueba. Ver preparacion.json y docs/particion_dataset.md.\n');
console.log(JSON.stringify({output: portable(path.relative(ROOT, output)), unique_images: retained.length,
  annotations: sum(totals), groups: groups.length, distribution: distributionRows,
  manifest_sha256: manifestDigest, class_balance_score: bestScore}, null, 2));
