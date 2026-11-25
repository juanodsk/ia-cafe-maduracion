import test from 'node:test';
import assert from 'node:assert/strict';
import { readAnalysisResponse } from '../src/api.js';

test('keeps a successful analysis response', async () => {
  const data = { detections: [], counts: {}, image: { data_url: 'data:image/jpeg;base64,example' } };
  assert.deepEqual(await readAnalysisResponse(Response.json(data)), data);
});

for (const status of [413, 429, 500, 502, 503, 504]) {
  test(`shows the HTTP cause for non-JSON error ${status}`, async () => {
    const response = new Response('<html>upstream unavailable</html>', { status });
    await assert.rejects(readAnalysisResponse(response), error =>
      error.message.includes(`HTTP ${status}`) && !error.message.includes('JSON') && !error.message.includes('<html>'));
  });
}

test('preserves API validation detail', async () => {
  await assert.rejects(readAnalysisResponse(Response.json({ detail: 'La foto supera 20 megapíxeles.' }, { status: 413 })), /20 megapíxeles.*HTTP 413/);
});

test('handles an empty gateway response', async () => {
  await assert.rejects(readAnalysisResponse(new Response('', { status: 502 })), /HTTP 502/);
});

test('rejects HTML returned with a successful status', async () => {
  await assert.rejects(readAnalysisResponse(new Response('<html>login</html>')), /respuesta inesperada/);
});

test('rejects an unrelated JSON response', async () => {
  await assert.rejects(readAnalysisResponse(Response.json({ status: 'ok' })), /incompleta/);
});
