import { useState, useRef } from 'react';
import { UploadCloud, FileType, Image as ImageIcon, Video, Mic } from 'lucide-react';
import { motion } from 'framer-motion';

export default function UploadArea({ modality, onFileSelect, onTextSubmit, loading }) {
  const [dragActive, setDragActive] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = function(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = function(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = function(e) {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Basic validation based on modality
    if (modality === 'image' && !file.type.startsWith('image/')) {
        alert("Please upload an image file");
        return;
    }
    if (modality === 'audio' && !file.type.startsWith('audio/')) {
        alert("Please upload an audio file");
        return;
    }
    if (modality === 'video' && !file.type.startsWith('video/')) {
        alert("Please upload a video file");
        return;
    }
    setSelectedFile(file);
    onFileSelect(file);
  };

  const getIcon = () => {
    switch(modality) {
      case 'image': return <ImageIcon size={64} className="upload-icon" />;
      case 'video': return <Video size={64} className="upload-icon" />;
      case 'audio': return <Mic size={64} className="upload-icon" />;
      default: return <UploadCloud size={64} className="upload-icon" />;
    }
  };

  if (modality === 'text') {
    return (
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="text-input-area"
      >
        <textarea 
          className="text-area glass"
          placeholder="Paste news article, social media post, or any text here for misinformation analysis..."
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          disabled={loading}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
          <button 
            className="btn btn-primary" 
            onClick={() => onTextSubmit(textInput)}
            disabled={loading || !textInput.trim()}
          >
            {loading ? <span className="loader"></span> : "Analyze Text"}
          </button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      style={{ width: '100%' }}
    >
      <div 
        className={`upload-area ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !loading && inputRef.current?.click()}
      >
        <input 
          ref={inputRef}
          type="file" 
          className="file-input" 
          onChange={handleChange}
          disabled={loading}
          accept={
            modality === 'image' ? 'image/*' : 
            modality === 'audio' ? 'audio/*' : 
            modality === 'video' ? 'video/*' : '*/*'
          }
        />
        
        {selectedFile && !loading ? (
          <div>
            <FileType size={48} className="upload-icon" style={{ color: 'var(--success-color)' }} />
            <h3>{selectedFile.name}</h3>
            <p>{(selectedFile.size / (1024*1024)).toFixed(2)} MB</p>
            <button className="btn btn-primary" onClick={(e) => { e.stopPropagation(); onFileSelect(selectedFile); }}>
              Analyze Now
            </button>
          </div>
        ) : loading ? (
           <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
             <span className="loader" style={{ width: '48px', height: '48px', borderWidth: '4px' }}></span>
             <h3>Analyzing {modality}...</h3>
             <p>Our AI is examining the structural and semantic composition.</p>
           </div>
        ) : (
          <>
            {getIcon()}
            <h3>Drag & Drop your {modality} or Click to Browse</h3>
            <p>We'll analyze it for synthetic artifacts and deepfake traces.</p>
          </>
        )}
      </div>
      
      {/* Previews for the selected file if we want them, optional */}
      {selectedFile && modality === 'image' && !loading && (
        <div style={{ textAlign: 'center' }}>
          <img src={URL.createObjectURL(selectedFile)} alt="Preview" className="preview-image" />
        </div>
      )}
      {selectedFile && modality === 'audio' && !loading && (
        <div style={{ textAlign: 'center' }}>
          <audio src={URL.createObjectURL(selectedFile)} controls className="audio-player" />
        </div>
      )}
      {selectedFile && modality === 'video' && !loading && (
        <div style={{ textAlign: 'center' }}>
          <video src={URL.createObjectURL(selectedFile)} controls className="video-player" />
        </div>
      )}
    </motion.div>
  );
}
