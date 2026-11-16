import axios from 'axios'
const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000' })
export const getHealth = () => api.get('/health')
export const getMetrics = () => api.get('/metrics')
export const getMachines = () => api.get('/machines')
export const getHistory = () => api.get('/history')
export const predictFailure = (data) => api.post('/predict', data)
