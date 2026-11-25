'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const ROOT = path.resolve(__dirname, '..');
const dataset = path.resolve(ROOT, process.argv[2] || 'dataset_preparado');
const report = JSON.parse(fs.readFileSync(path.join(dataset, 'preparacion.json'), 'utf8'));
const source = path.resolve(ROOT, report.source_directory);
const splits = ['train', 'valid', 'test'];
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
function readCsv(file) {
  const text = fs.readFileSync(file, 'utf8');
  const rows = [];
  let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && (ch === ',' || ch === '\n')) {
      row.push(cell); cell = '';
      if (ch === '\n') { rows.push(row); row = []; }
    } else if (ch !== '\r' || quoted) cell += ch;
  }
  assert(!quoted, 'Unclosed CSV quotation.');
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const headers = rows.shift();
  assert(new Set(headers).size === headers.length, 'Duplicate CSV headers.');
  return rows.map(values => {
    assert.equal(values.length, headers.length, 'CSV row column count mismatch.');
    return Object.fromEntries(headers.map((header, i) => [header, values[i]]));
  });
}
function uniqueMap(rows, key) {
  const result = new Map();
  for (const row of rows) {
    const value = key(row);
    assert(!result.has(value), 'Duplicate manifest key: ' + value);
    result.set(value, row);
  }
  return result;
}
function exclusive(map, key, split, kind) {
  if (map.has(key)) assert.equal(map.get(key), split, kind + ' crosses partitions: ' + key);
  else map.set(key, split);
}
function jpegDimensions(bytes) {
  assert(bytes[0] === 0xff && bytes[1] === 0xd8, 'Invalid JPEG signature.');
  let position = 2;
  while (position < bytes.length - 8) {
    assert.equal(bytes[position], 0xff, 'Invalid JPEG marker.');
    const marker = bytes[position + 1]; position += 2;
    if ([0xc0, 0xc1, 0xc2].includes(marker)) return {height: bytes.readUInt16BE(position + 3), width: bytes.readUInt16BE(position + 5)};
    if (marker === 0xda || marker === 0xd9) break;
    position += bytes.readUInt16BE(position);
  }
  assert.fail('JPEG dimensions not found.');
}

const imageRows = readCsv(path.join(dataset, 'csv', 'imagenes.csv'));
const annotationRows = readCsv(path.join(dataset, 'csv', 'anotaciones.csv'));
const groupRows = readCsv(path.join(dataset, 'csv', 'grupos.csv'));
const distribution = readCsv(path.join(dataset, 'csv', 'distribucion.csv'));
const classRows = readCsv(path.join(dataset, 'csv', 'distribucion_clases.csv'));
const duplicates = readCsv(path.join(dataset, 'csv', 'duplicados_excluidos.csv'));
const imagesByKey = uniqueMap(imageRows, r => r.split + ':' + r.image_id);
const annotationsByKey = uniqueMap(annotationRows, r => r.split + ':' + r.annotation_id);
const groupsByKey = uniqueMap(groupRows, r => r.group_id);
const sourceCoco = new Map();
let originalImages = 0, originalAnnotations = 0;
for (const sourceFile of report.source_annotation_files) {
  const data = fs.readFileSync(path.join(source, sourceFile.path));
  assert.equal(hash(data), sourceFile.sha256, 'Original source annotations changed.');
  const split = sourceFile.path.split('/')[0];
  const coco = JSON.parse(data);
  originalImages += coco.images.length;
  originalAnnotations += coco.annotations.length;
  sourceCoco.set(split, {images: uniqueMap(coco.images, r => r.id), annotations: uniqueMap(coco.annotations, r => r.id)});
}
const hashes = new Set(), assignedSources = new Set();
const sourceSplits = new Map(), captureSplits = new Map(), groupSplits = new Map();
const observed = [];
let totalAnnotations = 0;
for (const split of splits) {
  const dir = path.join(dataset, split);
  const coco = JSON.parse(fs.readFileSync(path.join(dir, '_annotations.coco.json'), 'utf8'));
  const images = uniqueMap(coco.images, r => r.id);
  uniqueMap(coco.annotations, r => r.id);
  const categories = uniqueMap(coco.categories, r => r.id);
  const counts = Object.fromEntries(report.class_labels.map(label => [label, 0]));
  const countsPerImage = new Map();
  for (const image of coco.images) {
    const row = imagesByKey.get(split + ':' + image.id);
    assert(row, 'Image missing from manifest.');
    assert.equal(row.relative_path, split + '/' + image.file_name);
    assert.equal(row.file_name, image.file_name);
    const src = sourceCoco.get(row.source_split).images.get(Number(row.source_image_id));
    assert(src, 'Manifest references missing source image.');
    assert.deepEqual({...image, id: src.id}, src, 'COCO image metadata changed.');
    const bytes = fs.readFileSync(path.join(dir, image.file_name));
    assert.equal(hash(bytes), row.sha256, 'Copied image hash mismatch.');
    assert.equal(hash(fs.readFileSync(path.join(source, row.source_path))), row.sha256, 'Source image changed.');
    assert.deepEqual(jpegDimensions(bytes), {width: image.width, height: image.height});
    assert(!hashes.has(row.sha256), 'Exact duplicate remains in prepared data.'); hashes.add(row.sha256);
    assert(!assignedSources.has(row.source_path), 'Source image assigned multiple times.'); assignedSources.add(row.source_path);
    exclusive(sourceSplits, image.file_name.split('.rf.')[0], split, 'Source name');
    const timestamp = image.file_name.match(/(\d{8})[_-](\d{6})(?!\d)/);
    if (timestamp) exclusive(captureSplits, timestamp[1] + '_' + timestamp[2], split, 'Capture timestamp');
    exclusive(groupSplits, row.group_id, split, 'Group');
    assert.equal(groupsByKey.get(row.group_id).split, split);
  }
  for (const annotation of coco.annotations) {
    assert(images.has(annotation.image_id), 'Broken COCO image reference.');
    assert(categories.has(annotation.category_id), 'Broken COCO category reference.');
    const row = annotationsByKey.get(split + ':' + annotation.id);
    assert(row, 'Annotation missing from CSV.');
    assert.equal(Number(row.image_id), annotation.image_id);
    const parent = imagesByKey.get(split + ':' + annotation.image_id);
    assert.equal(row.group_id, parent.group_id);
    assert.equal(row.source_split, parent.source_split);
    assert.equal(row.source_image_id, parent.source_image_id);
    const original = sourceCoco.get(row.source_split).annotations.get(Number(row.source_annotation_id));
    assert(original, 'Missing source annotation.');
    assert.equal(original.image_id, Number(parent.source_image_id));
    assert.deepEqual({...annotation, id: original.id, image_id: original.image_id}, original, 'Source geometry or annotation label changed.');
    const label = categories.get(annotation.category_id).name;
    assert.equal(label, row.label);
    assert.equal(Number(row.category_id), annotation.category_id);
    assert.deepEqual([row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height].map(Number), annotation.bbox);
    counts[label]++;
    const key = annotation.image_id + ':' + label;
    countsPerImage.set(key, (countsPerImage.get(key) || 0) + 1);
  }
  for (const image of coco.images) {
    const row = imagesByKey.get(split + ':' + image.id);
    let total = 0;
    for (const label of report.class_labels) {
      const n = countsPerImage.get(image.id + ':' + label) || 0;
      assert.equal(Number(row[label + '_annotations']), n);
      total += n;
    }
    assert.equal(Number(row.annotation_count), total);
  }
  const diskImages = fs.readdirSync(dir).filter(name => /\.jpg$/i.test(name));
  assert.equal(diskImages.length, coco.images.length, 'Extra or missing image files.');
  const dist = distribution.find(r => r.split === split);
  assert.equal(Number(dist.images), coco.images.length);
  assert.equal(Number(dist.annotations), coco.annotations.length);
  assert.equal(Number(dist.groups), [...groupSplits.values()].filter(s => s === split).length);
  const splitManifest = readCsv(path.join(dataset, 'csv', split + '.csv'));
  assert.deepEqual(splitManifest, imageRows.filter(r => r.split === split));
  for (const label of report.class_labels) {
    const classRow = classRows.find(r => r.split === split && r.label === label);
    assert.equal(Number(classRow.annotations), counts[label]);
    assert(counts[label] > 0, 'A maturity class is absent from a split.');
    const presence = coco.images.filter(im => countsPerImage.has(im.id + ':' + label)).length;
    assert.equal(Number(classRow.images_containing_class), presence);
  }
  const expected = report.split_counts.find(r => r.split === split);
  assert.equal(coco.images.length, expected.images);
  totalAnnotations += coco.annotations.length;
  observed.push({split, images: coco.images.length, groups: Number(dist.groups), annotations: coco.annotations.length});
}
for (const group of groupRows) assert.equal(imageRows.filter(r => r.group_id === group.group_id).length, Number(group.images));
for (const duplicate of duplicates) {
  assert(assignedSources.has(duplicate.kept_source_path));
  assert(!assignedSources.has(duplicate.excluded_source_path));
  assert.equal(hash(fs.readFileSync(path.join(source, duplicate.kept_source_path))), duplicate.sha256);
  assert.equal(hash(fs.readFileSync(path.join(source, duplicate.excluded_source_path))), duplicate.sha256);
}
assert.equal(originalImages, report.original_images);
assert.equal(hashes.size, report.retained_images);
assert.equal(imageRows.length, hashes.size);
assert.equal(totalAnnotations, report.retained_annotations);
assert.equal(annotationRows.length, totalAnnotations);
assert.equal(groupSplits.size, report.groups);
assert.equal(duplicates.length, report.excluded_duplicate_files);
assert.equal(originalImages, hashes.size + duplicates.length);
assert.equal(originalAnnotations, totalAnnotations + duplicates.reduce((n, r) => n + Number(r.excluded_annotations), 0));
assert.equal(hash(fs.readFileSync(path.join(dataset, 'csv', 'imagenes.csv'))), report.image_manifest_sha256);
const rawTargets = [0.70, 0.15, 0.15].map(r => r * hashes.size);
const targetImages = rawTargets.map(Math.floor);
const priority = [0, 1, 2].sort((a, b) => (rawTargets[b] - targetImages[b]) - (rawTargets[a] - targetImages[a]) || a - b);
for (let remaining = hashes.size - targetImages.reduce((a, b) => a + b, 0), i = 0; remaining > 0; i++, remaining--) targetImages[priority[i]]++;
assert.deepEqual(observed.map(r => r.images), targetImages, 'Prepared split does not match 70/15/15 rounded to integer images.');
console.log(JSON.stringify({status: 'OK', original_images_preserved: originalImages, prepared_images: hashes.size,
  annotations: totalAnnotations, groups: groupSplits.size, partitions: observed,
  exact_duplicate_files: 0, source_names_crossing_splits: 0, capture_timestamps_crossing_splits: 0,
  group_ids_crossing_splits: 0, coco_and_csv_traceability: 'verified'}, null, 2));
