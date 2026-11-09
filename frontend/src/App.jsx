
import { useEffect, useMemo, useState } from 'react'
import {
  Activity, BarChart3, Bell, ChevronRight, CircleHelp, Cpu, Factory,
  FileImage, Gauge, LayoutDashboard, Menu, Settings2, ShieldCheck,
  UploadCloud, Wifi, X, Zap, Trash2, RefreshCw
} from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from 'recharts'
import { getHealth, getModelStatus, inspectImage } from './api'

const MAX_FILES = 25
const MAX_FILE_SIZE = 12 * 1024 * 1024
const COLORS = ['#35c6b0', '#4da3ff', '#a78bfa', '#f6b94a', '#f08080', '#8babc7']

const navItems = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'inspection', label: 'AI Inspection', icon: FileImage },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'history', label: 'Inspection History', icon: Activity },
]

const pretty = (s = '') =>
  s.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase())

function getTopPredictions(probabilities = {}) {
  return Object.entries(probabilities)
    .map(([name, probability]) => ({ name, probability: Number(probability) }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 3)
}

function getEntropy(probabilities = {}) {
  const values = Object.values(probabilities)
    .map(Number)
    .filter(p => Number.isFinite(p) && p >= 0)

  const total = values.reduce((sum, p) => sum + p, 0)
  if (!values.length || total <= 0) return null

  // Normalize defensively so small floating-point differences do not distort entropy.
  const normalized = values.map(p => p / total).filter(p => p > 0)
  return -normalized.reduce((sum, p) => sum + p * Math.log2(p), 0)
}

function getEntropyLabel(entropy) {
  if (entropy === null || !Number.isFinite(entropy)) return 'Unavailable'
  if (entropy < 0.5) return 'Low uncertainty'
  if (entropy < 1.5) return 'Moderate uncertainty'
  return 'High uncertainty'
}

function getItemEntropy(item) {
  return getEntropy(item.probabilities || item.result?.probabilities || {})
}

function Sidebar({ page, setPage, mobileOpen, close }) {
  return (
    <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
      <div className="brand">
        <div className="brand-mark"><Factory size={21} /></div>
        <div>
          <b>STEELVISION</b>
          <span>INTELLIGENCE PLATFORM</span>
        </div>
        <button className="mobile-close" onClick={close}>
          <X size={18} />
        </button>
      </div>

      <div className="workspace-label">WORKSPACE</div>

      <nav>
        {navItems.map(item => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? 'active' : ''}`}
              onClick={() => {
                setPage(item.id)
                close()
              }}
            >
              <Icon size={18} />
              <span>{item.label}</span>
              {page === item.id && <ChevronRight size={15} className="nav-arrow" />}
            </button>
          )
        })}
      </nav>

      <div className="sidebar-bottom">
        <div className="system-card">
          <div className="system-card-top">
            <span className="pulse-dot" /> SYSTEM STATUS
          </div>
          <b>Inspection engine</b>
          <small>Local inference node</small>
        </div>

        <button className="nav-item muted">
          <Settings2 size={18} /> Settings
        </button>
        <button className="nav-item muted">
          <CircleHelp size={18} /> Documentation
        </button>

        <div className="sidebar-version">
          STEELVISION AI <span>v2.1.0</span>
        </div>
      </div>
    </aside>
  )
}

function Header({ title, subtitle, onMenu, apiOnline }) {
  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onMenu}>
        <Menu size={20} />
      </button>

      <div>
        <div className="breadcrumb">
          WORKSPACE <ChevronRight size={13} /> <b>{title.toUpperCase()}</b>
        </div>
        <div className="top-subtitle">{subtitle}</div>
      </div>

      <div className="top-actions">
        <div className={`connection ${apiOnline ? 'online' : 'offline'}`}>
          <span /> {apiOnline ? 'API CONNECTED' : 'API OFFLINE'}
        </div>
        <button className="icon-button" title="Notifications">
          <Bell size={18} />
        </button>
        <div className="avatar">SV</div>
      </div>
    </header>
  )
}

function Metric({ label, value, note, icon: Icon, accent = 'teal' }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <div className={`metric-icon ${accent}`}>
          <Icon size={18} />
        </div>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  )
}

function HistoryTable({ rows }) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>IMAGE</th>
            <th>PREDICTED CLASS</th>
            <th>CONFIDENCE</th>
            <th>INFERENCE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.id}>
              <td>
                <span className="table-file">
                  <FileImage size={15} /> {row.filename}
                </span>
              </td>
              <td>
                <span className="class-pill">
                  {pretty(row.predicted_class)}
                </span>
              </td>
              <td><b>{(row.confidence * 100).toFixed(1)}%</b></td>
              <td>{row.inference_time_ms} ms</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Overview({ model, history, apiOnline, setPage }) {
  const counts = useMemo(() => {
    const out = Object.fromEntries((model?.classes || []).map(c => [c, 0]))
    history.forEach(h => {
      if (h.predicted_class in out) out[h.predicted_class]++
    })
    return out
  }, [history, model])

  const chartData = Object.entries(counts).map(([name, count]) => ({
    name: pretty(name),
    count,
  }))

  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">OPERATIONS / COMMAND CENTER</div>
          <h1>Inspection overview</h1>
          <p>Monitor steel surface quality and AI inspection activity.</p>
        </div>
        <button className="primary-button" onClick={() => setPage('inspection')}>
          <UploadCloud size={17} /> Open inspection studio
        </button>
      </div>

      <div className="metrics-grid">
        <Metric label="TOTAL INSPECTIONS" value={history.length}
          note="Images inspected this session" icon={FileImage} />
        <Metric label="MODEL ARCHITECTURE" value="ResNet-34"
          note="Six-class classifier" icon={Cpu} accent="blue" />
        <Metric label="AI ENGINE" value={apiOnline ? 'Online' : 'Offline'}
          note="FastAPI inference service" icon={Wifi}
          accent={apiOnline ? 'teal' : 'amber'} />
        <Metric label="DEFECT CLASSES" value={model?.classes?.length ?? '—'}
          note="NEU surface defect categories" icon={ShieldCheck} accent="purple" />
      </div>

      <div className="content-grid">
        <section className="panel chart-panel">
          <div className="panel-heading">
            <div>
              <h2>Inspection distribution</h2>
              <p>Predicted classes from this session</p>
            </div>
            <span className="panel-tag">LIVE SESSION</span>
          </div>

          {history.length ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -18, bottom: 35 }}>
                  <CartesianGrid stroke="#253340" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" angle={-20} textAnchor="end"
                    interval={0} tick={{ fill: '#8495a5', fontSize: 10 }} />
                  <YAxis allowDecimals={false} tick={{ fill: '#8495a5', fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[5, 5, 0, 0]}>
                    {chartData.map((entry, i) => (
                      <Cell key={entry.name} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty-chart">
              <div className="empty-icon"><BarChart3 size={24} /></div>
              <b>No inspection data yet</b>
              <span>Run an inspection to populate this chart.</span>
            </div>
          )}
        </section>

        <section className="panel model-panel">
          <div className="panel-heading">
            <div><h2>Model configuration</h2><p>Active inference configuration</p></div>
          </div>
          <div className="config-row"><span>Architecture</span><b>ResNet-34</b></div>
          <div className="config-row"><span>Dataset</span><b>NEU Surface Defect</b></div>
          <div className="config-row"><span>Input resolution</span><b>224 × 224</b></div>
          <div className="config-row"><span>Checkpoint</span>
            <b className={model?.checkpoint_present ? 'text-green' : 'text-amber'}>
              {model?.checkpoint_present ? 'Detected' : 'Not found'}
            </b>
          </div>
        </section>
      </div>

      <section className="panel recent-panel">
        <div className="panel-heading">
          <div><h2>Recent inspections</h2><p>Latest predictions</p></div>
          <button className="text-button" onClick={() => setPage('history')}>
            View history <ChevronRight size={15} />
          </button>
        </div>
        {history.length
          ? <HistoryTable rows={history.slice(0, 5)} />
          : <div className="empty-row">No inspections recorded yet.</div>}
      </section>

      <div className="disclaimer">
        <ShieldCheck size={16} />
        Research prototype. Predictions require human review before industrial use.
      </div>
    </div>
  )
}

/*
  IMPORTANT:
  This component stays mounted even when another page is selected.
  Its state is also controlled by the parent App.
*/
function Inspection({
  currentItems,
  setCurrentItems,
  onResults,
  onNewInspection,
}) {
  const [files, setFiles] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')

  const selected = currentItems.find(item => item.id === selectedId)
  const selectedResult = selected?.result

  function addFiles(fileList) {
    const incoming = Array.from(fileList || [])
    const valid = []
    const errors = []

    incoming.forEach(file => {
      if (!/\.(jpg|jpeg|png|bmp)$/i.test(file.name)) {
        errors.push(`${file.name}: unsupported file format`)
      } else if (file.size > MAX_FILE_SIZE) {
        errors.push(`${file.name}: exceeds 12 MB`)
      } else {
        valid.push(file)
      }
    })

    const remaining = MAX_FILES - files.length
    const accepted = valid.slice(0, remaining)

    if (valid.length > remaining) {
      errors.push(`Only ${MAX_FILES} images are allowed per batch.`)
    }

    setFiles(prev => [...prev, ...accepted])
    setError(errors.join('. '))
  }

  function removeFile(index) {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  async function runBatch() {
    if (!files.length || busy) return

    setBusy(true)
    setProgress(0)
    setError('')

    const completed = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]

      try {
        const data = await inspectImage(file)

        const item = {
          id: `${Date.now()}-${i}-${Math.random().toString(36).slice(2)}`,
          filename: file.name,
          result: data,
          predicted_class: data.predicted_class,
          confidence: data.confidence,
          inference_time_ms: data.inference_time_ms,
          // Keep probabilities at the history-item level for Analytics.
          probabilities: data.probabilities || {},
          created_at: new Date().toISOString(),
        }

        completed.push(item)
        onResults([item])
        setCurrentItems(prev => [...prev, item])
        setSelectedId(item.id)
      } catch (e) {
        completed.push({
          id: `${Date.now()}-${i}`,
          filename: file.name,
          error: e.message || 'Inspection failed',
          result: null,
        })
      }

      setProgress(Math.round(((i + 1) / files.length) * 100))
    }

    const failures = completed.filter(item => item.error)

    if (failures.length) {
      setError(`${failures.length} image(s) could not be inspected.`)
    }

    setFiles([])
    setBusy(false)
  }

  function startNewInspection() {
    if (busy) return
    setFiles([])
    setSelectedId(null)
    setError('')
    setProgress(0)
    onNewInspection()
  }

  const probs = selectedResult?.probabilities
    ? Object.entries(selectedResult.probabilities)
      .map(([name, value]) => ({ name, value: Number(value) * 100 }))
      .sort((a, b) => b.value - a.value)
    : []

  const top3 = selectedResult
    ? getTopPredictions(selectedResult.probabilities)
    : []

  const entropy = selectedResult
    ? getEntropy(selectedResult.probabilities || {})
    : null

  const classCount = Object.keys(selectedResult?.probabilities || {}).length || 6
  const maxEntropy = Math.log2(classCount)
  const normalizedEntropy = entropy !== null && maxEntropy > 0
    ? Math.min(1, entropy / maxEntropy)
    : null

  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">QUALITY CONTROL / IMAGE ANALYSIS</div>
          <h1>AI inspection studio</h1>
          <p>Inspect individual images or process a batch of up to 25 images.</p>
        </div>
        <button className="primary-button" onClick={startNewInspection} disabled={busy}>
          <RefreshCw size={16} /> New Inspection
        </button>
      </div>

      <div className="inspection-layout">
        <section className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <h2>Batch image input</h2>
              <p>JPG, PNG, BMP · Max 12 MB per image · Up to 25 images</p>
            </div>
            <FileImage size={19} className="muted-icon" />
          </div>

          <label
            className="dropzone"
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              e.preventDefault()
              addFiles(e.dataTransfer.files)
            }}
          >
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.bmp"
              multiple
              onChange={e => {
                addFiles(e.target.files)
                e.target.value = ''
              }}
            />
            <div className="upload-symbol"><UploadCloud size={26} /></div>
            <b>Drop steel images here</b>
            <span>or click to browse multiple files</span>
            <small>{files.length} / {MAX_FILES} images selected</small>
          </label>

          {files.length > 0 && (
            <div className="batch-file-list">
              <div className="batch-list-heading">
                <b>Selected images</b>
                <span>{files.length} files</span>
              </div>

              {files.map((file, index) => (
                <div className="batch-file-row" key={`${file.name}-${index}`}>
                  <FileImage size={15} />
                  <span>{file.name}</span>
                  <small>{(file.size / 1024).toFixed(0)} KB</small>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => removeFile(index)}
                    title="Remove image"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {busy && (
            <div className="batch-progress">
              <div>
                <span>Batch processing</span>
                <b>{progress}%</b>
              </div>
              <div className="batch-progress-track">
                <div style={{ width: `${progress}%` }} />
              </div>
              <small>Images are being processed sequentially.</small>
            </div>
          )}

          <button
            className="primary-button run-button"
            disabled={!files.length || busy}
            onClick={runBatch}
          >
            {busy
              ? <><span className="spinner" /> Processing batch...</>
              : <><Zap size={17} /> Run Batch Inspection ({files.length})</>}
          </button>

          {error && <div className="error-box">{error}</div>}
        </section>

        <section className="panel result-panel">
          <div className="panel-heading">
            <div>
              <h2>Inspection result</h2>
              <p>Select a completed image to view its analysis.</p>
            </div>
            <span className="panel-tag">OUTPUT</span>
          </div>

          {selectedResult ? (
            <>
              <div className="prediction-result">
                <div>
                  <span className="result-label">PREDICTED DEFECT</span>
                  <h2>{pretty(selectedResult.predicted_class)}</h2>
                </div>
                <div className="confidence">
                  <b>{(selectedResult.confidence * 100).toFixed(1)}%</b>
                  <span>CONFIDENCE</span>
                </div>
              </div>

              <div className="prob-title">TOP 3 PREDICTIONS</div>

              {top3.map((p, i) => (
                <div className="prob-row" key={p.name}>
                  <div>
                    <span>{pretty(p.name)}</span>
                    <b>{(p.probability * 100).toFixed(2)}%</b>
                  </div>
                  <div className="prob-track">
                    <div
                      style={{
                        width: `${p.probability * 100}%`,
                        background: COLORS[i],
                      }}
                    />
                  </div>
                </div>
              ))}

              <div className="uncertainty-card">
                <div>
                  <span>Prediction entropy</span>
                  <b>{entropy === null ? 'Unavailable' : `${entropy.toFixed(3)} bits`}</b>
                </div>
                <div>
                  <span>{getEntropyLabel(entropy)}</span>
                  <b>{normalizedEntropy === null ? 'No probability data' : `${(normalizedEntropy * 100).toFixed(1)}% of maximum`}</b>
                </div>
              </div>
              {entropy !== null && entropy >= 1.5 && (
                <div className="uncertainty-review-alert" role="status">
                  <b>Human review recommended</b>
                  <span>The model is uncertain between multiple defect classes. Verify this image before making a quality decision.</span>
                </div>
              )}

              <div className="inference-meta">
                <span>Inference time</span>
                <b>{selectedResult.inference_time_ms} ms</b>
              </div>
            </>
          ) : (
            <div className="empty-result">
              <div className="empty-icon"><Gauge size={25} /></div>
              <b>Awaiting image analysis</b>
              <span>Upload images and run an inspection to see the model output.</span>
            </div>
          )}
        </section>
      </div>

      {currentItems.length > 0 && (
        <section className="panel batch-results-panel">
          <div className="panel-heading">
            <div>
              <h2>Batch results</h2>
              <p>{currentItems.length} completed inspection(s) in this workspace</p>
            </div>
            <span className="panel-tag">CURRENT SESSION</span>
          </div>

          <div className="batch-result-list">
            {currentItems.map(item => (
              <button
                type="button"
                key={item.id}
                className={`batch-result-item ${selectedId === item.id ? 'selected' : ''}`}
                onClick={() => setSelectedId(item.id)}
              >
                <FileImage size={16} />
                <span>{item.filename}</span>
                {item.error ? (
                  <b className="batch-failed">Failed</b>
                ) : (
                  <>
                    <small>{pretty(item.predicted_class)}</small>
                    <b>{(item.confidence * 100).toFixed(1)}%</b>
                  </>
                )}
              </button>
            ))}
          </div>
        </section>
      )}

      {selectedResult && (
        <>
          <section className="panel probability-panel">
            <div className="panel-heading">
              <div>
                <h2>Full class probability distribution</h2>
                <p>Softmax probabilities across all six defect categories.</p>
              </div>
            </div>

            {probs.map((p, i) => (
              <div className="prob-row" key={p.name}>
                <div>
                  <span>{pretty(p.name)}</span>
                  <b>{p.value.toFixed(2)}%</b>
                </div>
                <div className="prob-track">
                  <div style={{ width: `${p.value}%`, background: COLORS[i] }} />
                </div>
              </div>
            ))}
          </section>

          <section className="panel cam-panel">
            <div className="panel-heading">
              <div>
                <h2>Visual explainability</h2>
                <p>Grad-CAM highlights regions contributing to the predicted class.</p>
              </div>
              <span className="panel-tag">GRAD-CAM</span>
            </div>

            <div className="cam-grid">
              <div>
                <img src={selectedResult.original_image} alt="Preprocessed original" />
                <span>PREPROCESSED INPUT</span>
              </div>
              <div>
                <img src={selectedResult.gradcam_overlay} alt="Grad-CAM overlay" />
                <span>ACTIVATION OVERLAY</span>
              </div>
            </div>

            <div className="cam-note">
              {selectedResult.note || 'Grad-CAM is an approximate model explanation, not a precise defect boundary.'}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

function Analytics({ history, model }) {
  const classes = model?.classes || []
  const data = classes.map(name => ({
    name: pretty(name),
    count: history.filter(h => h.predicted_class === name).length,
  }))

  const averageConfidence = history.length
    ? history.reduce((sum, item) => sum + item.confidence, 0) / history.length
    : 0

  const averageTime = history.length
    ? history.reduce((sum, item) => sum + item.inference_time_ms, 0) / history.length
    : 0

  const entropyValues = history
    .map(getItemEntropy)
    .filter(value => value !== null)
  const averageEntropy = entropyValues.length
    ? entropyValues.reduce((sum, value) => sum + value, 0) / entropyValues.length
    : null
  const averageNormalizedEntropy = averageEntropy !== null
    ? Math.min(1, averageEntropy / Math.log2(6))
    : null
  const highUncertaintyCount = entropyValues.filter(value => value >= 1.5).length

  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">QUALITY INTELLIGENCE / REPORTING</div>
          <h1>Inspection analytics</h1>
          <p>Explore predictions generated during this session.</p>
        </div>
      </div>

      <div className="metrics-grid">
        <Metric label="IMAGES ANALYZED" value={history.length}
          note="Current browser session" icon={FileImage} />
        <Metric label="UNIQUE CLASSES"
          value={new Set(history.map(x => x.predicted_class)).size}
          note="Classes observed" icon={BarChart3} accent="blue" />
        <Metric label="AVG. CONFIDENCE"
          value={history.length ? `${(averageConfidence * 100).toFixed(1)}%` : '—'}
          note="Not model accuracy" icon={Gauge} accent="purple" />
        <Metric label="AVG. INFERENCE"
          value={history.length ? `${averageTime.toFixed(0)} ms` : '—'}
          note="Observed processing time" icon={Zap} accent="amber" />
      </div>

      <div className="metrics-grid">
        <Metric label="AVG. ENTROPY"
          value={averageEntropy !== null ? `${averageEntropy.toFixed(3)} bits` : '—'}
          note={`${entropyValues.length} inspection(s) with probability data`} icon={Activity} />
        <Metric label="AVG. UNCERTAINTY"
          value={averageNormalizedEntropy !== null ? `${(averageNormalizedEntropy * 100).toFixed(1)}%` : '—'}
          note="Normalized entropy, not defect risk" icon={Activity} accent="blue" />
        <Metric label="HUMAN REVIEW FLAGS"
          value={highUncertaintyCount}
          note="Entropy ≥ 1.5 bits, provisional threshold" icon={ShieldCheck} accent="amber" />
        <Metric label="HIGH CONFIDENCE"
          value={history.filter(x => x.confidence >= 0.9).length}
          note="Predictions ≥ 90%" icon={ShieldCheck} accent="blue" />
        <Metric label="LOW CONFIDENCE"
          value={history.filter(x => x.confidence < 0.6).length}
          note="Predictions below 60%" icon={Gauge} accent="amber" />
        <Metric label="TOTAL PROCESSING"
          value={`${(history.reduce((sum, x) => sum + x.inference_time_ms, 0) / 1000).toFixed(2)} s`}
          note="Combined inference time" icon={Zap} accent="purple" />
      </div>

      <section className="panel chart-panel analytics-chart">
        <div className="panel-heading">
          <div>
            <h2>Predicted defect frequency</h2>
            <p>Count of predictions by class</p>
          </div>
        </div>

        {history.length ? (
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data}>
                <CartesianGrid stroke="#253340" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#8495a5', fontSize: 11 }} />
                <YAxis allowDecimals={false} tick={{ fill: '#8495a5', fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[5, 5, 0, 0]}>
                  {data.map((d, i) => (
                    <Cell key={d.name} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="empty-row">Run inspections to generate analytics.</div>
        )}
      </section>

      <div className="disclaimer">
        <ShieldCheck size={16} />
        Confidence and entropy are model-output statistics, not verified accuracy.
        Accuracy, precision, recall and F1 require ground-truth labels.
      </div>
    </div>
  )
}

function History({ history }) {
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">QUALITY CONTROL / RECORDS</div>
          <h1>Inspection history</h1>
          <p>All completed predictions in this browser session.</p>
        </div>
      </div>

      <section className="panel recent-panel">
        <div className="panel-heading">
          <div>
            <h2>Inspection records</h2>
            <p>{history.length} image(s) inspected this session</p>
          </div>
        </div>
        {history.length
          ? <HistoryTable rows={history} />
          : <div className="empty-row">No inspection records yet.</div>}
      </section>

      <div className="disclaimer">
        <Activity size={16} />
        History remains available while this application session is running.
        No database or persistent storage is used.
      </div>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState('overview')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [apiOnline, setApiOnline] = useState(false)
  const [model, setModel] = useState(null)

  // Shared inspection workspace state.
  const [currentItems, setCurrentItems] = useState([])
  const [history, setHistory] = useState([])

  useEffect(() => {
    getHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false))

    getModelStatus()
      .then(setModel)
      .catch(() => setModel(null))
  }, [])

  function addResults(items) {
    setHistory(prev => [...items, ...prev])
  }

  const pageMeta = {
    overview: ['Inspection overview', 'Steel surface quality monitoring'],
    inspection: ['AI inspection studio', 'Image classification and explainability'],
    analytics: ['Inspection analytics', 'Operational prediction insights'],
    history: ['Inspection history', 'Inspection records and review'],
  }[page]

  return (
    <div className="app-shell">
      <Sidebar
        page={page}
        setPage={setPage}
        mobileOpen={mobileOpen}
        close={() => setMobileOpen(false)}
      />

      {mobileOpen && (
        <div className="mobile-scrim" onClick={() => setMobileOpen(false)} />
      )}

      <main className="main-area">
        <Header
          title={pageMeta[0]}
          subtitle={pageMeta[1]}
          onMenu={() => setMobileOpen(true)}
          apiOnline={apiOnline}
        />

        {page === 'overview' && (
          <Overview
            model={model}
            history={history}
            apiOnline={apiOnline}
            setPage={setPage}
          />
        )}

        {/*
          Keep Inspection mounted even when hidden.
          This preserves selected image and current workspace state.
        */}
        <div style={{ display: page === 'inspection' ? 'block' : 'none' }}>
          <Inspection
            currentItems={currentItems}
            setCurrentItems={setCurrentItems}
            onResults={addResults}
            onNewInspection={() => setCurrentItems([])}
          />
        </div>

        {page === 'analytics' && <Analytics history={history} model={model} />}
        {page === 'history' && <History history={history} />}

        <footer className="footer">
          STEELVISION AI <span>•</span>
          DEEP LEARNING STEEL SURFACE INSPECTION <span>•</span>
          RESEARCH PROTOTYPE
        </footer>
      </main>
    </div>
  )
}
