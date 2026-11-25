import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const stages = [
  { key: 'unripe', label: 'Verde', color: '#42c665' },
  { key: 'semi_ripe', label: 'Pintón', color: '#f0c23b' },
  { key: 'ripe', label: 'Maduro', color: '#f05252' },
  { key: 'overripe', label: 'Sobremaduro', color: '#c66bbb' },
  { key: 'dry', label: 'Seco', color: '#bc8659' },
  { key: 'unknown', label: 'Incierto', color: '#a1aaa8' },
];
const stage = (key) => stages.find(s => s.key === key) || stages[5];
const models = {
  efficientnet_b0: 'EfficientNet B0 · recomendado', resnet50: 'ResNet50',
  densenet121: 'DenseNet121', vit_b_16: 'ViT B/16', swin_t: 'Swin T',
  ensemble: 'Ensemble · los 5 modelos',
};
const percent = (score) => new Intl.NumberFormat('es-CO', { style: 'percent', maximumFractionDigits: 1 }).format(score);

function Icon({ name, size = 22, ...props }) {
  const paths = {
    upload: <><path d="M12 16V4m-4 4 4-4 4 4"/><path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/></>,
    photo: <><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 5-5 4 4 4-6 5 7"/></>,
    scan: <><path d="M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4M7 12h10"/></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4M4 17v3h16v-3"/></>,
    leaf: <><path d="M20 3C9 2 3 6 4 13c1 7 12 7 15-3l1-7Z"/><path d="M4 21c0-6 5-10 10-13"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    close: <path d="m6 6 12 12M6 18 18 6"/>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [model, setModel] = useState('efficientnet_b0');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState(null);
  const [example, setExample] = useState(false);
  const [deployment, setDeployment] = useState(null);
  const input = useRef(null);
  const request = useRef(null);
  const version = useRef(0);

  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/health', { signal: controller.signal }).then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setDeployment(data.deployment); }).catch(() => {});
    return () => controller.abort();
  }, []);

  function choose(next, isExample = false) {
    request.current?.abort();
    version.current += 1;
    setBusy(false); setResult(null); setSelected(null); setError('');
    if (!next) return;
    if (next.size > 15 * 1024 * 1024) { setError('La foto supera 15 MB. Elige un archivo más pequeño.'); return; }
    if (!/\.(jpe?g|png|webp)$/i.test(next.name)) { setError('Selecciona una foto JPG, PNG o WEBP.'); return; }
    setFile(next); setExample(isExample);
  }

  async function analyze(photo = file, useExample = example) {
    if (!photo) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const id = ++version.current;
    setBusy(true); setError(''); setResult(null); setSelected(null);
    try {
      const body = new FormData();
      body.append('file', photo); body.append('model', model);
      const response = await fetch('/api/analyze', { method: 'POST', body, signal: controller.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'No se pudo analizar la foto.');
      if (version.current === id) {
        setResult({ ...data, example: useExample });
        setSelected(data.detections[0]?.id ?? null);
      }
    } catch (e) {
      if (e.name !== 'AbortError' && version.current === id) setError(e.message === 'Failed to fetch' ? 'No hay conexión con el servidor local. Revisa que esté encendido.' : e.message);
    } finally {
      if (version.current === id) setBusy(false);
    }
  }

  async function tryExample() {
    setError(''); setBusy(true);
    const id = ++version.current;
    try {
      const response = await fetch('/api/example');
      if (!response.ok) throw new Error('La foto de ejemplo no está disponible.');
      const photo = new File([await response.blob()], 'ejemplo_validacion.jpg', { type: 'image/jpeg' });
      if (version.current !== id) return;
      choose(photo, true);
      await analyze(photo, true);
    } catch (e) {
      if (version.current === id) { setError(e.message); setBusy(false); }
    }
  }

  function download(kind) {
    const anchor = document.createElement('a');
    const stem = (file?.name || 'cafe').replace(/\.[^.]+$/, '');
    let href;
    if (kind === 'image') {
      const bytes = Uint8Array.from(atob(result.annotated_image.split(',')[1]), c => c.charCodeAt(0));
      href = URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
    }
    else {
      const { image, annotated_image, ...report } = result;
      href = URL.createObjectURL(new Blob([JSON.stringify({ ...report, image_size: { width: image.width, height: image.height }, source_file: file.name }, null, 2)], { type: 'application/json' }));
    }
    anchor.href = href; anchor.download = `${stem}_resultado.${kind === 'image' ? 'jpg' : 'json'}`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(href), 60000);
  }

  const imageSource = result?.image.data_url || preview;

  return <>
    <header className="topbar"><div className="shell topbar-inner">
      <a className="brand" href="/" aria-label="Café, inicio"><span className="brand-mark"><Icon name="leaf" size={25}/></span><span>café<span className="brand-dot">.</span><small>ANÁLISIS DE MADUREZ</small></span></a>
      <span className="local-badge"><span/> {deployment === 'local' ? 'Procesamiento local' : 'Análisis en el servidor'}</span>
    </div></header>
    <main className="shell">
      <section className="intro"><div><p className="eyebrow">DEL FRUTO A LA COSECHA</p><h1>La madurez, a simple vista.</h1><p>Sube una foto y reconoce el estado de cada fruto de café.</p></div><span className="project-label">Visión artificial<br/><strong>5 estados de madurez</strong></span></section>
      <div className="workspace">
        <aside className="controls card">
          <div className="section-label"><span className="step">01</span><h2>Tu fotografía</h2></div>
          <input ref={input} className="file-input" id="photo-upload" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" disabled={busy} onChange={e => { choose(e.target.files?.[0]); e.target.value = ''; }}/>
          <button className={`dropzone ${dragging ? 'dragging' : ''}`} disabled={busy} onClick={() => input.current?.click()} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); if (!busy) choose(e.dataTransfer.files?.[0]); }}>
            <span className="upload-icon"><Icon name={file ? 'photo' : 'upload'} size={27}/></span>
            <strong>{file ? 'Cambiar fotografía' : 'Selecciona una foto'}</strong>
            <span>o arrástrala hasta aquí</span><small>JPG, PNG o WEBP · hasta 15 MB</small>
          </button>
          {file && <div className="file-detail"><Icon name="check" size={17}/><span title={file.name}>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(1)} MB</small></div>}
          <div className="model-field"><label htmlFor="model">Modelo de clasificación</label><select id="model" value={model} disabled={busy} onChange={e => { setModel(e.target.value); setResult(null); setSelected(null); }}>{Object.entries(models).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select><p>{model === 'ensemble' ? 'Combina los cinco modelos. Puede tardar más al cargar.' : 'Detecta los frutos y clasifica cada uno por separado.'}</p></div>
          <button className="primary" disabled={!file || busy} onClick={() => analyze()}>{busy ? <span className="spinner"/> : <Icon name="scan" size={20}/>} {busy ? 'Analizando fotografía…' : 'Analizar madurez'}</button>
          <div className="example-action"><span>¿Quieres verlo en acción?</span><button disabled={busy} onClick={tryExample}>Probar con una foto de ejemplo</button><small>Foto del conjunto de validación.</small></div>
          {error && <div className="error" role="alert">{error}</div>}
          <div className="privacy"><Icon name="leaf" size={18}/><p>{deployment === 'local' ? 'La foto se procesa en este equipo y no se guarda automáticamente.' : 'La foto se envía al servidor de la aplicación para analizarla y no se guarda automáticamente.'}</p></div>
        </aside>
        <section className="analysis card" aria-label="Análisis de la fotografía" aria-busy={busy}>
          <div className="analysis-heading"><div className="section-label"><span className="step">02</span><h2>Lectura de madurez</h2></div><span className={`state ${result ? 'ready' : ''}`}>{busy ? 'Procesando' : result ? 'Análisis completo' : file ? 'Lista para analizar' : 'Esperando fotografía'}</span></div>
          {imageSource ? <div className="photo-board">
            <div className="photo-container"><img className="main-photo" src={imageSource} alt={result ? 'Café analizado con recuadros numerados por fruto' : 'Vista previa de la foto seleccionada'}/>
              {result && <svg className="boxes" viewBox={`0 0 ${result.image.width} ${result.image.height}`} aria-label="Frutos detectados">{result.detections.map(d => <g key={d.id} role="button" tabIndex="0" aria-label={`Fruto ${d.id}: ${d.label}, ${percent(d.score)}`} className={`fruit-box ${selected === d.id ? 'selected' : ''}`} style={{ '--box-color': stage(d.class).color }} onClick={() => { setSelected(d.id); document.getElementById(`fruit-${d.id}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(d.id); document.getElementById(`fruit-${d.id}`)?.scrollIntoView({ block: 'nearest' }); } }}><title>#{d.id} · {d.label} · {percent(d.score)}</title><rect x={d.bbox[0]} y={d.bbox[1]} width={d.bbox[2] - d.bbox[0]} height={d.bbox[3] - d.bbox[1]} rx="3"/><rect className="number-bg" x={d.bbox[0]} y={Math.max(0, d.bbox[1] - 23)} width="27" height="23" rx="3"/><text x={d.bbox[0] + 13.5} y={Math.max(0, d.bbox[1] - 23) + 16} textAnchor="middle">{d.id}</text></g>)}</svg>}
            </div>
            {busy && <div className="processing-overlay"><span className="spinner"/><strong>Reconociendo los frutos</strong><span>La primera carga del modelo puede tardar unos segundos.</span></div>}
          </div> : <div className="empty-board"><div className="viewfinder"><Icon name="scan" size={46}/></div><h3>Cada fruto cuenta una historia.</h3><p>Tu foto aparecerá aquí con los frutos<br/>marcados según su madurez.</p><div className="empty-dots">{stages.slice(0, 5).map(s => <i key={s.key} style={{ background: s.color }}/>)}</div></div>}
          <div className="legend" aria-label="Colores de madurez">{stages.map(s => <span key={s.key}><i style={{ background: s.color }}/>{s.label}</span>)}</div>
          {!result && <div className="before-analysis"><Icon name="scan" size={19}/><p>{busy ? 'Detectando frutos y calculando su estado de madurez…' : file ? 'Todo listo. Pulsa «Analizar madurez» para ver los resultados.' : 'Para una mejor lectura, usa una foto nítida con los frutos visibles.'}</p></div>}
          {result && <div className="results" aria-live="polite">
            <div className="result-title"><div><h3>{result.total} {result.total === 1 ? 'fruto detectado' : 'frutos detectados'}</h3><p>{result.model_label} · {(result.processing_ms / 1000).toLocaleString('es-CO', { maximumFractionDigits: 1 })} s · {result.device}</p></div><div className="download-actions"><button className="secondary" onClick={() => download('image')}><Icon name="download" size={18}/> Descargar foto</button><button className="text-button" onClick={() => download('json')}>JSON</button></div></div>
            {result.example && <p className="sample-note">Ejemplo del conjunto de validación; sirve para demostrar el funcionamiento.</p>}
            {result.total > 0 ? <><div className="count-chips">{stages.filter(s => result.counts[s.key]).map(s => <span key={s.key}><i style={{ background: s.color }}/>{s.label}<strong>{result.counts[s.key]}</strong></span>)}</div><div className="fruit-table"><div className="table-head"><span>Fruto / madurez</span><span>Puntuación del modelo</span></div><div className="fruit-list">{result.detections.map(d => <button id={`fruit-${d.id}`} key={d.id} className={`fruit-row ${selected === d.id ? 'active' : ''}`} onClick={() => setSelected(d.id)} aria-pressed={selected === d.id}><span className="fruit-name"><b className="fruit-number">{String(d.id).padStart(2, '0')}</b><i style={{ background: stage(d.class).color }}/><span>{d.label}{d.class === 'unknown' && <small>Posible: {d.candidate_label}</small>}</span></span><span className="score-cell"><span className="score-track"><span style={{ width: `${d.score * 100}%`, background: stage(d.class).color }}/></span><strong>{percent(d.score)}</strong></span></button>)}</div></div></> : <p className="no-fruit">No se detectaron frutos. Prueba una foto más cercana, bien iluminada y sin desenfoque.</p>}
            <p className="score-note">{result.score_note}</p>
          </div>}
        </section>
      </div>
      <footer><span>Proyecto de clasificación de café</span><span>3 CNN + 2 Transformers · Detector de frutos</span></footer>
    </main>
  </>;
}

createRoot(document.getElementById('root')).render(<App/>);
