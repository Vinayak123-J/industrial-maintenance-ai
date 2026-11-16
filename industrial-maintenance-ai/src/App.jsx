import { useEffect, useState } from 'react'
import { Activity, AlertTriangle, Cpu, Gauge, RefreshCw, ShieldCheck, Wrench } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { getHealth, getMetrics, getMachines, getHistory, predictFailure } from './api'

const initial = { type: 'M', air_temperature_k: 300, process_temperature_k: 310, rotational_speed_rpm: 1500, torque_nm: 40, tool_wear_min: 120 }
const pct = (v) => `${(Number(v || 0) * 100).toFixed(1)}%`

export default function App() {
  const [health, setHealth] = useState(null), [metrics, setMetrics] = useState(null)
  const [machines, setMachines] = useState([]), [history, setHistory] = useState([])
  const [form, setForm] = useState(initial), [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false), [error, setError] = useState('')

  async function refresh() {
    try {
      const [h, m, ms, hs] = await Promise.all([getHealth(), getMetrics(), getMachines(), getHistory()])
      setHealth(h.data); setMetrics(m.data); setMachines(ms.data); setHistory(hs.data); setError('')
    } catch { setError('Cannot connect to the API. Start the FastAPI backend on port 8000.') }
  }
  useEffect(() => { refresh() }, [])
  const update = (e) => setForm({ ...form, [e.target.name]: e.target.value })
  async function submit(e) {
    e.preventDefault(); setBusy(true); setError(''); setResult(null)
    try { const response = await predictFailure({ ...form, air_temperature_k: +form.air_temperature_k, process_temperature_k: +form.process_temperature_k, rotational_speed_rpm: +form.rotational_speed_rpm, torque_nm: +form.torque_nm, tool_wear_min: +form.tool_wear_min }); setResult(response.data); await refresh() }
    catch (err) { setError(err.response?.data?.detail || 'Prediction failed. Check that the model has been trained.') }
    finally { setBusy(false) }
  }
  const cards = [
    { label: 'Registered machines', value: machines.length || '—', icon: Cpu },
    { label: 'Model accuracy', value: metrics?.accuracy != null ? pct(metrics.accuracy) : '—', icon: Gauge },
    { label: 'Model recall', value: metrics?.recall != null ? pct(metrics.recall) : '—', icon: ShieldCheck },
    { label: 'Saved predictions', value: history.length, icon: Activity }
  ]
  const chartData = history.slice(0, 8).reverse().map((r, i) => ({ name: `P${i + 1}`, risk: +(r.failure_probability * 100).toFixed(1) }))
  return <div className="shell">
    <aside className="sidebar"><div className="brand"><div className="brandIcon"><Wrench size={22}/></div><div><b>INDUSTRIAL AI</b><small>MAINTENANCE PLATFORM</small></div></div><div className="nav active"><Activity size={18}/> Overview</div><div className="nav"><Cpu size={18}/> Machine health</div><div className="nav"><Gauge size={18}/> Predictive analysis</div><div className="sideBottom"><span className={`dot ${health?.status === 'ok' ? '' : 'off'}`}></span> API {health?.status === 'ok' ? 'Connected' : 'Disconnected'}</div></aside>
    <main className="main"><header><div><p className="eyebrow">OPERATIONS / OVERVIEW</p><h1>Predictive Maintenance Dashboard</h1><p className="muted">Monitor machine health and predict potential equipment failures.</p></div><button className="refresh" onClick={refresh}><RefreshCw size={16}/> Refresh</button></header>
      {error && <div className="error">{error}</div>}
      <section className="stats">{cards.map(({label,value,icon:Icon})=><div className="stat" key={label}><div className="statTop"><span>{label}</span><Icon size={19}/></div><strong>{value}</strong><small>Live project data</small></div>)}</section>
      <section className="grid"><div className="panel"><div className="panelHead"><div><h2>Machine fleet</h2><p className="muted">Current sample equipment register</p></div><span className="tag">LIVE API</span></div><div className="tableWrap"><table><thead><tr><th>Machine ID</th><th>Type</th><th>Area</th><th>Status</th></tr></thead><tbody>{machines.map(m=><tr key={m.machine_id}><td><b>{m.machine_id}</b></td><td>{m.type}</td><td>{m.area}</td><td><span className="status">{m.status}</span></td></tr>)}</tbody></table></div></div>
      <div className="panel"><div className="panelHead"><div><h2>Model performance</h2><p className="muted">Held-out test dataset</p></div></div>{metrics?.accuracy != null ? <div className="metricList">{[['Accuracy',metrics.accuracy],['Precision',metrics.precision],['Recall',metrics.recall],['F1 score',metrics.f1_score],['ROC AUC',metrics.roc_auc]].map(([k,v])=><div className="metric" key={k}><span>{k}</span><b>{pct(v)}</b><div className="track"><i style={{width:pct(v)}}/></div></div>)}</div> : <p className="muted">Train the model to load performance metrics.</p>}</div></section>
      <section className="grid lower"><div className="panel"><div className="panelHead"><div><h2>Failure risk estimator</h2><p className="muted">Enter sensor readings to estimate machine failure probability.</p></div><AlertTriangle size={20}/></div>
        <form className="form" onSubmit={submit}><label>Machine type<select name="type" value={form.type} onChange={update}><option value="L">L - Low</option><option value="M">M - Medium</option><option value="H">H - High</option></select></label>
        <label>Air temperature (K)<input name="air_temperature_k" type="number" step="0.1" value={form.air_temperature_k} onChange={update} required/></label>
        <label>Process temperature (K)<input name="process_temperature_k" type="number" step="0.1" value={form.process_temperature_k} onChange={update} required/></label>
        <label>Rotational speed (RPM)<input name="rotational_speed_rpm" type="number" value={form.rotational_speed_rpm} onChange={update} required/></label>
        <label>Torque (Nm)<input name="torque_nm" type="number" step="0.1" value={form.torque_nm} onChange={update} required/></label>
        <label>Tool wear (minutes)<input name="tool_wear_min" type="number" value={form.tool_wear_min} onChange={update} required/></label>
        <button className="primary" disabled={busy}>{busy ? 'Analyzing...' : 'Predict machine failure'}</button></form>
        {result && <div className={`result ${result.risk_level.toLowerCase()}`}><div className="resultHead"><b>{result.risk_level} risk</b><strong>{pct(result.failure_probability)}</strong></div><p>Predicted failure: <b>{result.predicted_failure ? 'Yes' : 'No'}</b></p><p>{result.recommendation}</p></div>}
      </div><div className="panel"><div className="panelHead"><div><h2>Recent prediction history</h2><p className="muted">Failure probability from latest API predictions</p></div></div>{chartData.length ? <div className="chart"><ResponsiveContainer width="100%" height="230"><BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name"/><YAxis domain={[0,100]} unit="%"/><Tooltip formatter={(v)=>`${v}%`}/><Bar dataKey="risk" name="Failure risk" fill="#3978d5" radius={[5,5,0,0]}/></BarChart></ResponsiveContainer></div> : <div className="empty">Run a prediction to populate the history chart.</div>}
        <div className="history">{history.slice(0,5).map(r=><div className="historyRow" key={r.id}><span>Prediction #{r.id}</span><b>{pct(r.failure_probability)}</b><em className={r.risk_level.toLowerCase()}>{r.risk_level}</em></div>)}</div></div></section>
      <footer><span>AI-Based Industrial Maintenance and Process Optimization System</span><span>Educational prototype · Random Forest</span></footer>
    </main>
  </div>
}
