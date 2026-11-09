const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) throw new Error('Backend is not responding')
  return response.json()
}

export async function getModelStatus() {
  const response = await fetch(`${API_BASE}/model-status`)
  if (!response.ok) throw new Error('Could not retrieve model status')
  return response.json()
}

export async function inspectImage(file) {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${API_BASE}/inspect`, { method: 'POST', body: form })
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || 'Inspection failed')
  return data
}
