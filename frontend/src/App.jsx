import { useState } from 'react';
import axios from 'axios';
import { Shield, FileText, Image as ImageIcon, Video, Mic } from 'lucide-react';
import UploadArea from './components/UploadArea';
import ResultDetails from './components/ResultDetails';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1/detect';

function App() {
  const [modality, setModality] = useState('image'); // text, image, audio, video
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleTextSubmit = async (text) => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API_BASE_URL}/text`, { text });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = async (file) => {
    setLoading(true);
    setError(null);
    
    // Set mock progress or processing text? Handled by loading state
    const formData = new FormData();
    formData.append("file", file);

    try {
      const endpoint = `${API_BASE_URL}/${modality}`;
      const res = await axios.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred while uploading or analyzing the file.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="layout-container">
      <header className="header glass glass-panel">
        <div className="logo">
          <Shield className="logo-icon" size={32} />
          <h1>TruthLens <span className="gradient-text">AI</span></h1>
        </div>
        <div style={{ color: 'var(--text-muted)' }}>
          "Truth Across Every Medium"
        </div>
      </header>

      <main>
        {!result ? (
          <>
            <div className="modality-tabs">
              <button 
                className={`modality-tab ${modality === 'text' ? 'active' : ''}`}
                onClick={() => setModality('text')}
                disabled={loading}
              >
                <FileText className="modality-icon" />
                <span>Text Analysis</span>
              </button>
              
              <button 
                className={`modality-tab ${modality === 'image' ? 'active' : ''}`}
                onClick={() => setModality('image')}
                disabled={loading}
              >
                <ImageIcon className="modality-icon" />
                <span>Image Forensics</span>
              </button>
              
              <button 
                className={`modality-tab ${modality === 'audio' ? 'active' : ''}`}
                onClick={() => setModality('audio')}
                disabled={loading}
              >
                <Mic className="modality-icon" />
                <span>Audio Deepfake</span>
              </button>
              
              <button 
                className={`modality-tab ${modality === 'video' ? 'active' : ''}`}
                onClick={() => setModality('video')}
                disabled={loading}
              >
                <Video className="modality-icon" />
                <span>Video Analysis</span>
              </button>
            </div>

            <div className="glass glass-panel upload-container">
              {error && (
                <div style={{ background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger-color)', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                  Error: {error}
                </div>
              )}
              
              <UploadArea 
                modality={modality} 
                onFileSelect={handleFileSelect} 
                onTextSubmit={handleTextSubmit}
                loading={loading}
              />
            </div>
          </>
        ) : (
          <ResultDetails result={result} onReset={handleReset} />
        )}
      </main>

      <footer style={{ marginTop: '4rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        <p>&copy; {new Date().getFullYear()} TruthLens AI. Enterprise Misinformation Detection System.</p>
      </footer>
    </div>
  );
}

export default App;
