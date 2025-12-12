// src/api.js
import axios from "axios";

// Your FastAPI is running here:
const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export async function fetchModels() {
  const res = await api.get("/models");
  return res.data; // { model_id: { name, file }, ... }
}

export async function fetchFeatures() {
  const res = await api.get("/features");
  return res.data.features; // [ "radius_mean", ... ]
}

export async function fetchMetrics() {
  const res = await api.get("/metrics");
  return res.data; // { model_id: { name, accuracy, ... }, ... }
}

export async function predict(modelId, data) {
  const res = await api.post(
    "/predict",
    { data }, // matches BreastCancerInput(data: Dict[str,float])
    { params: { model_id: modelId } }
  );
  return res.data; // { prediction, probabilities, ... }
}
export async function fetchFeatureRanges() {
  const res = await fetch("http://127.0.0.1:8000/feature-ranges");
  if (!res.ok) throw new Error("Failed to fetch feature ranges");
  return await res.json();
}
export async function fetchDatasetInfo() {
  const res = await api.get("/dataset-info");
  return res.data;
}

export async function fetchEvaluationTable() {
  const res = await api.get("/evaluation-table");
  return res.data; // array of rows
}
