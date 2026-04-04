import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, ShieldAlert, Cpu } from 'lucide-react';

export default function ResultDetails({ result, onReset }) {
  if (!result) return null;

  const score = result.authenticity_score;
  
  let scoreClass = "score-authentic";
  let statusText = "Likely Authentic";
  let StatusIcon = CheckCircle;
  let summary = "No significant traces of synthetic generation or misinformation detected.";

  if (score < 40) {
    scoreClass = "score-fake";
    statusText = "High Risk: Synthetic or Misinformation";
    StatusIcon = AlertTriangle;
    summary = "System detected strong indicators of manipulation or synthetic creation.";
  } else if (score < 80) {
    scoreClass = "score-warning";
    statusText = "Warning: Anomalies Detected";
    StatusIcon = ShieldAlert;
    summary = "Mixed signals found. The content may be heavily edited, compressed, or misleading.";
  }

  // Create radial gradient for the score circle percentage
  const circleStyle = {
    '--percentage': `${score}%`
  };

  return (
    <div className="results-container">
      <div className="results-grid">
        <motion.div 
          className="glass glass-panel score-card"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4 }}
        >
          <div className={`score-circle ${scoreClass}`} style={circleStyle}>
            {score.toFixed(0)}<span style={{ fontSize: '1.5rem' }}>%</span>
          </div>
          <h2 className={`score-label ${scoreClass}`}>{statusText}</h2>
          <p style={{ color: 'var(--text-muted)' }}>Authenticity Score</p>
          
          <div style={{ marginTop: '2rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
            <Cpu size={16} />
            <span>AI Confidence: {result.confidence_score}%</span>
          </div>
          
          <button className="btn btn-outline" style={{ marginTop: '2rem', width: '100%' }} onClick={onReset}>
            Analyze Another
          </button>
        </motion.div>

        <motion.div 
          className="glass glass-panel details-card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <h3>Detailed Analysis</h3>
          <div className="explanation-text">
            {result.explanation}
          </div>
          
          <h3><Cpu size={18} /> Model Telemetry</h3>
          <div className="metrics-grid">
            {Object.entries(result.details || {}).map(([key, value]) => {
              // format key
              const formattedKey = key.replace(/_/g, ' ');
              return (
                <div key={key} className="metric-item">
                  <div className="metric-label">{formattedKey}</div>
                  <div className="metric-value">
                    {typeof value === 'object' ? JSON.stringify(value) : value.toString()}
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
