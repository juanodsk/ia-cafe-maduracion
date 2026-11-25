// A proxy or Uvicorn may return plain text/HTML on failure, rather than JSON.
export function analysisHttpError(status) {
  const messages = {
    404: 'No se encontró la API de análisis. Revisa el dominio y el despliegue.',
    413: 'El servidor rechazó la foto por su tamaño. Usa un archivo de hasta 15 MB.',
    422: 'El servidor no pudo leer la foto. Prueba con otro archivo JPG, PNG o WEBP.',
    429: 'Hay otro análisis en curso. Espera a que termine y vuelve a intentarlo.',
    500: 'El servidor no pudo completar el análisis. Revisa los registros del servicio app.',
    502: 'El servidor de análisis no está disponible o se está reiniciando. Espera unos segundos y vuelve a intentarlo.',
    503: 'El servidor está ocupado o temporalmente no disponible. Espera unos segundos y vuelve a intentarlo.',
    504: 'El servidor tardó demasiado en analizar la foto. Prueba con una imagen más cercana o de menor tamaño.',
  };
  return `${messages[status] || 'No se pudo completar el análisis.'} (HTTP ${status})`;
}

export async function readAnalysisResponse(response) {
  const body = await response.text();
  let data;
  try {
    data = JSON.parse(body);
  } catch {
    if (!response.ok) throw new Error(analysisHttpError(response.status));
    throw new Error(`El servidor devolvió una respuesta inesperada. Recarga la página y vuelve a intentarlo. (HTTP ${response.status})`);
  }
  if (!response.ok) {
    throw new Error(typeof data?.detail === 'string'
      ? `${data.detail} (HTTP ${response.status})` : analysisHttpError(response.status));
  }
  if (!Array.isArray(data?.detections) || !data?.image?.data_url || !data?.counts) {
    throw new Error('La respuesta del análisis está incompleta. Recarga la página y vuelve a intentarlo.');
  }
  return data;
}
