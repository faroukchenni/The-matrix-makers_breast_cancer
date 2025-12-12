import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Activity, TrendingUp, AlertCircle, CheckCircle } from "lucide-react";
import {
  fetchModels,
  fetchFeatures,
  fetchMetrics,
  fetchFeatureRanges,
  predict,
} from "./api";
import "./index.css";

const PIE_COLORS = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6"];

function App() {
  const [models, setModels] = useState({});
  const [metrics, setMetrics] = useState({});
  const [features, setFeatures] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [patientData, setPatientData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [featureRanges, setFeatureRanges] = useState({});
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [evalTable, setEvalTable] = useState([]);

  // Load everything from backend on mount
  useEffect(() => {
    async function init() {
      try {
        const [m, feat, met, ranges] = await Promise.all([
          fetchModels(),
          fetchFeatures(),
          fetchMetrics(),
          fetchFeatureRanges(),
          fetchDatasetInfo(),
          fetchEvaluationTable(),
        ]);

        setModels(m || {});
        setFeatures(feat || []);
        setMetrics(met || {});
        setFeatureRanges(ranges || {});
        setDatasetInfo(ds || {});
        setEvalTable(ev || []);

        const ids = Object.keys(m || {});
        if (ids.length) setSelectedModelId(ids[0]);

        const defaults = {};
        (feat || []).forEach((f) => (defaults[f] = 0));
        setPatientData(defaults);
        setErrorMsg("");
      } catch (err) {
        console.error(err);
        setErrorMsg("Could not load models / features / metrics from backend.");
      }
    }

    init();
  }, []);

  const modelIds = Object.keys(models);
  const selectedMetrics = selectedModelId ? metrics[selectedModelId] : null;
  const selectedModelName =
    (selectedModelId &&
      (metrics[selectedModelId]?.name || models[selectedModelId]?.name)) ||
    "Selected model";

  const handleFeatureChange = (feat, value) => {
    setPatientData((prev) => ({
      ...prev,
      [feat]: parseFloat(value) || 0,
    }));
  };

  const handlePredict = async () => {
    if (!selectedModelId) return;
    try {
      setLoading(true);
      setErrorMsg("");
      const res = await predict(selectedModelId, patientData);
      setPrediction(res);
    } catch (err) {
      console.error(err);
      setPrediction(null);
      setErrorMsg("Error calling prediction API.");
    } finally {
      setLoading(false);
    }
  };

  // ===== Mean/Std realistic auto-fill (Gaussian) =====
  const autoFillFromDistribution = () => {
    if (!features.length) return;

    if (!featureRanges || Object.keys(featureRanges).length === 0) {
      setErrorMsg("Feature ranges not loaded. Check /feature-ranges in FastAPI.");
      return;
    }

    const newData = {};

    for (const feat of features) {
      const stats = featureRanges[feat];

      if (!stats || stats.mean == null || stats.std == null) {
        newData[feat] = 0;
        continue;
      }

      // Box–Muller transform to sample standard normal
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

      let value = stats.mean + z0 * stats.std;

      // Clamp within observed min/max
      if (stats.min != null) value = Math.max(stats.min, value);
      if (stats.max != null) value = Math.min(stats.max, value);

      // Format nicely
      newData[feat] =
        Math.abs(value) < 10
          ? Number(value.toFixed(4))
          : Number(value.toFixed(2));
    }

    setPatientData(newData);
    setPrediction(null);
    setErrorMsg("");
  };

  const comparisonData = modelIds
    .filter((id) => metrics[id] && !metrics[id].error)
    .map((id) => ({
      model: metrics[id].name || models[id].name || id,
      accuracy: (metrics[id]?.accuracy || 0) * 100,
      f1: (metrics[id]?.f1 || 0) * 100,
    }));

  const confusionData = selectedMetrics
    ? [
        { name: "True Negatives", value: selectedMetrics.tn ?? 0 },
        { name: "False Positives", value: selectedMetrics.fp ?? 0 },
        { name: "False Negatives", value: selectedMetrics.fn ?? 0 },
        { name: "True Positives", value: selectedMetrics.tp ?? 0 },
      ]
    : [];

  const radarData = selectedMetrics
    ? [
        { metric: "Accuracy", value: (selectedMetrics.accuracy || 0) * 100 },
        { metric: "Precision", value: (selectedMetrics.precision || 0) * 100 },
        { metric: "Recall", value: (selectedMetrics.recall || 0) * 100 },
        { metric: "F1-Score", value: (selectedMetrics.f1 || 0) * 100 },
        { metric: "AUC", value: (selectedMetrics.auc || 0) * 100 },
      ]
    : [];

  return (
    <div className="app-root">
      <div className="app-container">
        {/* HEADER */}
        <header className="app-header">
          <div>
            <h1 className="header-title">
              <Activity size={28} />
              Breast Cancer Prediction Dashboard
            </h1>
            <p className="header-subtitle">
              Multi-model diagnostic classifier – FastAPI backend, React frontend.
            </p>
          </div>
          <div style={{ textAlign: "right", fontSize: 12 }}>
            <div>
              Models deployed: <strong>{modelIds.length}</strong>
            </div>
            <div>
              Features: <strong>{features.length}</strong>
            </div>
          </div>
        </header>

        {errorMsg && (
          <div
            style={{
              marginTop: 10,
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #fecaca",
              background: "#fef2f2",
              color: "#b91c1c",
              fontSize: 13,
            }}
          >
            {errorMsg}
          </div>
        )}

        {/* MODEL SELECTION + METRICS */}
        <section className="card">
          <h2 style={{ fontSize: 18, margin: 0, marginBottom: 6 }}>Select model</h2>
          <div className="model-card-grid">
            {modelIds.map((id) => {
              const m = metrics[id] || {};
              const selected = id === selectedModelId;
              return (
                <div
                  key={id}
                  className={`model-card ${selected ? "selected" : ""}`}
                  onClick={() => setSelectedModelId(id)}
                >
                  <div className="model-card-name">
                    {m.name || models[id].name || id}
                  </div>
                  <div className="model-card-acc">
                    {m.accuracy ? (m.accuracy * 100).toFixed(1) : "--"}%
                  </div>
                  <div className="model-card-label">Accuracy</div>
                </div>
              );
            })}
          </div>

          {selectedMetrics && !selectedMetrics.error && (
            <>
              <h3 style={{ marginTop: 12, marginBottom: 4, fontSize: 16 }}>
                {selectedModelName} – key metrics
              </h3>
              <div className="metric-grid">
                {[
                  { label: "Accuracy", key: "accuracy", icon: CheckCircle, color: "#2563eb" },
                  { label: "Precision", key: "precision", icon: TrendingUp, color: "#16a34a" },
                  { label: "Recall", key: "recall", icon: Activity, color: "#7c3aed" },
                  { label: "F1-score", key: "f1", icon: AlertCircle, color: "#ec4899" },
                ].map((entry) => {
                  const Icon = entry.icon;
                  const v = selectedMetrics[entry.key] ?? 0;
                  return (
                    <div className="metric-card" key={entry.label}>
                      <div className="metric-title-row">
                        <span>{entry.label}</span>
                        <Icon size={18} color={entry.color} />
                      </div>
                      <div className="metric-value">{(v * 100).toFixed(1)}%</div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </section>

        {/* MAIN GRID: LEFT = PATIENT FORM, RIGHT = CHARTS */}
        <div className="main-grid">
          {/* LEFT: PATIENT FEATURES */}
          <section className="card">
            <h2 style={{ fontSize: 18, margin: 0 }}>Patient features & prediction</h2>
            <p style={{ fontSize: 12, color: "#6b7280", marginTop: 4 }}>
              Click “Auto-fill realistic patient” then run prediction.
            </p>

            <div className="feature-grid">
              {features.map((feat) => (
                <div key={feat} className="feature-field">
                  <label>{feat}</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={patientData[feat] ?? 0}
                    onChange={(e) => handleFeatureChange(feat, e.target.value)}
                  />
                </div>
              ))}
            </div>

            {/* BUTTON ROW (secondary + primary) */}
            <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
              <button
                className="secondary-btn"
                onClick={autoFillFromDistribution}
                disabled={!features.length}
                type="button"
              >
                Auto-fill realistic patient
              </button>

              <button
                className="primary-btn"
                onClick={handlePredict}
                disabled={loading || !selectedModelId}
                type="button"
              >
                {loading ? "Predicting..." : "Run prediction"}
              </button>
            </div>

            {prediction && (
              <div>
                <div
                  className={
                    "result-box " +
                    (prediction.prediction === 1 ? "result-bad" : "result-ok")
                  }
                >
                  {prediction.prediction === 1 ? (
                    <>
                      ⚠ Predicted class: <strong>Malignant (1)</strong>
                    </>
                  ) : (
                    <>
                      ✅ Predicted class: <strong>Benign (0)</strong>
                    </>
                  )}
                </div>

                {prediction.probabilities &&
                  prediction.probabilities.length === 2 && (
                    <div style={{ marginTop: 8, fontSize: 12 }}>
                      <div>
                        Benign probability:{" "}
                        {(prediction.probabilities[0] * 100).toFixed(1)}%
                      </div>
                      <div>
                        Malignant probability:{" "}
                        {(prediction.probabilities[1] * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}

                <p style={{ fontSize: 11, color: "#6b7280", marginTop: 6 }}>
                  This is a machine learning demo. It is not a medical diagnosis.
                </p>
              </div>
            )}
          </section>

          {/* RIGHT: CHARTS */}
          <section
            className="card"
            style={{ display: "flex", flexDirection: "column", gap: 16 }}
          >
            <div style={{ height: 230 }}>
              <h3 style={{ margin: 0, marginBottom: 6, fontSize: 16 }}>
                {selectedModelName} – confusion summary
              </h3>
              {confusionData.some((d) => d.value > 0) ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={confusionData}
                      dataKey="value"
                      nameKey="name"
                      outerRadius={80}
                      label
                    >
                      {confusionData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={PIE_COLORS[index % PIE_COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p style={{ fontSize: 12, color: "#6b7280" }}>
                  Confusion matrix values not available for this model.
                </p>
              )}
            </div>

            <div style={{ height: 230 }}>
              <h3 style={{ margin: 0, marginBottom: 6, fontSize: 16 }}>
                Model performance comparison
              </h3>
              {comparisonData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="model"
                      angle={-15}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis domain={[80, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy (%)" />
                    <Bar dataKey="f1" fill="#ec4899" name="F1-score (%)" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p style={{ fontSize: 12, color: "#6b7280" }}>
                  Not enough metrics to display comparison yet.
                </p>
              )}
            </div>
          </section>
        </div>

        {/* BOTTOM METRICS SUMMARY */}
        <section className="card" style={{ marginTop: 12 }}>
          <h3 style={{ margin: 0, marginBottom: 6, fontSize: 16 }}>
            Metrics overview – {selectedModelName}
          </h3>
          {radarData.length ? (
            <div style={{ height: 230 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={radarData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#a855f7" name="Score (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p style={{ fontSize: 12, color: "#6b7280" }}>
              Metrics not available for this model. Make sure{" "}
              <code>metrics.json</code> was exported.
            </p>
          )}
        </section>

        <div className="footer">
          Dataset: Wisconsin Breast Cancer. Models trained in notebook, exported as artifacts,
          served via FastAPI, visualized with React + Recharts.
        </div>
      </div>
    </div>
  );
}

export default App;
