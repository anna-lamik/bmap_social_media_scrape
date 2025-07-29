const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function analyze(payload) {
  const res = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error('Analysis failed');
  return res.json();
}

export default { analyze };
