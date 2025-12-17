// src/api.js
import axios from "axios";

// Always talk to backend via IPv4 to avoid Windows IPv6 localhost weirdness.
const BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
});

// ✅ attach JWT token automatically if present
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export async function fetchModels() {
  const res = await api.get("/models");
  return res.data;
}

export async function fetchFeatures() {
  const res = await api.get("/features");
  return res.data.features;
}

export async function fetchMetrics() {
  const res = await api.get("/metrics");
  return res.data;
}

export async function fetchFeatureRanges() {
  const res = await api.get("/feature-ranges");
  return res.data;
}

export async function predict(modelId, data) {
  const res = await api.post(
    "/predict",
    { data },
    { params: { model_id: modelId } }
  );
  return res.data;
}

// Future endpoints (safe to keep)
export async function fetchDatasetInfo() {
  const res = await api.get("/dataset-info");
  return res.data;
}

export async function fetchEvaluationTable() {
  const res = await api.get("/evaluation-table");
  return res.data;
}

export async function fetchMonitoringSummary() {
  const res = await fetch(`${BASE_URL}/monitoring/summary`);
  if (!res.ok) throw new Error("Failed to load monitoring summary");
  return res.json();
}

// ✅ fixed: API_BASE was undefined
export async function fetchMonitoringPing() {
  const res = await api.get("/monitoring/ping");
  return res.data;
}

export async function fetchLivePredictions(limit = 10) {
  const res = await fetch(`${BASE_URL}/monitoring/live?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to fetch live predictions");
  return res.json();
}

export async function chat(messages) {
  const base = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

  const res = await fetch(`${base}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages, // ✅ EXACTLY what FastAPI expects
    }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || "Chat failed");
  return data.reply;
}

// ✅ Auth
function getBase() {
  return import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
}

export async function signup(email, password, role) {
  const res = await fetch(`${getBase()}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, role }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || "Signup failed");
  return data;
}

export async function login(email, password) {
  const res = await fetch(`${getBase()}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || "Login failed");
  return data;
}
