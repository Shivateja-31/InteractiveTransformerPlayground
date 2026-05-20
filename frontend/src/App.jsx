import React, { useState, useEffect, useRef } from 'react';
import PipelineDiagram from './components/PipelineDiagram';
import AttentionHeatmap from './components/AttentionHeatmap';
import TokenProbabilities from './components/TokenProbabilities';
import EmbeddingVisualizer from './components/EmbeddingVisualizer';
import { Activity, Cpu, Sparkles, AlertCircle } from 'lucide-react';

export default function App() {
  const [inputText, setInputText] = useState("First, you know Caius Marcius");
  const [temperature, setTemperature] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  
  const handlePredict = async () => {
    if (!inputText) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, temperature })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAppendToken = () => {
    if (result && result.predicted_token) {
      const token = result.predicted_token;
      // Add a space if the input doesn't already end with one, unless input is empty
      const prefix = inputText.length > 0 && !inputText.endsWith(' ') ? ' ' : '';
      setInputText(prev => prev + prefix + token);
    }
  };

  useEffect(() => {
    // Initial prediction on load
    handlePredict();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen bg-background text-textPrimary flex flex-col font-sans selection:bg-primary/30">
      
      {/* Header */}
      <header className="border-b border-white/5 bg-surface/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded bg-primary/20 flex items-center justify-center border border-primary/50 text-primary shadow-[0_0_15px_rgba(59,130,246,0.3)]">
              <Sparkles size={18} />
            </div>
            <h1 className="font-bold text-lg tracking-tight">Interactive Transformer Playground</h1>
          </div>
          
          <div className="flex items-center space-x-4 text-sm text-textSecondary">
            <div className="flex items-center">
              <Cpu size={14} className="mr-1.5" />
              <span>MiniGPT (Custom)</span>
            </div>
            <div className="flex items-center">
              <Activity size={14} className="mr-1.5 text-green-500" />
              <span>API Online</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column - Input & Control */}
        <div className="lg:col-span-4 space-y-6 flex flex-col h-full">
          
          <div className="bg-surface p-6 rounded-xl border border-white/5 flex-shrink-0">
            <h3 className="text-sm font-semibold text-textSecondary uppercase tracking-wider mb-4 flex justify-between items-center">
              <span>Input Sequence</span>
              <span className="text-xs font-normal opacity-70 bg-white/5 px-2 py-1 rounded">Word-level</span>
            </h3>
            <textarea
              className="w-full bg-background border border-white/10 rounded-lg p-3 h-32 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary resize-none font-mono text-sm leading-relaxed"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter some text..."
            />
            
            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center space-x-2 w-1/2">
                <span className="text-xs text-textSecondary w-8">Temp:</span>
                <input 
                  type="range" 
                  min="0.1" max="2.0" step="0.1" 
                  value={temperature} 
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-primary h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer" 
                />
                <span className="text-xs text-textSecondary font-mono">{temperature.toFixed(1)}</span>
              </div>
              
              <button 
                onClick={handlePredict}
                disabled={loading}
                className="bg-primary hover:bg-primaryHover text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center shadow-[0_0_15px_rgba(59,130,246,0.3)] hover:shadow-[0_0_20px_rgba(59,130,246,0.5)]"
              >
                {loading ? 'Processing...' : 'Predict Next'}
              </button>
            </div>
            
            {error && (
              <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-sm flex items-start">
                <AlertCircle size={16} className="mr-2 mt-0.5 flex-shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </div>
          
          {/* Prediction Result Block */}
          {result && (
            <div className="bg-surface p-6 rounded-xl border border-white/5 flex flex-col border-l-2 border-l-accent flex-shrink-0">
              <h3 className="text-sm font-semibold text-textSecondary uppercase tracking-wider mb-2">Predicted Output</h3>
              <div className="flex items-end justify-between mt-2">
                <div className="font-mono text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent truncate max-w-[200px]" title={result.predicted_token}>
                  {result.predicted_token}
                </div>
                <button 
                  onClick={handleAppendToken}
                  className="text-xs text-textPrimary bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded transition-colors"
                >
                  Append to Input
                </button>
              </div>
            </div>
          )}

          {/* Probabilities */}
          <div className="flex-1 min-h-[250px]">
             <TokenProbabilities topTokens={result?.top_tokens} />
          </div>

        </div>
        
        {/* Middle Column - Pipeline Diagram */}
        <div className="lg:col-span-3">
          <PipelineDiagram />
        </div>

        {/* Right Column - Visualizations */}
        <div className="lg:col-span-5 flex flex-col space-y-6 h-full">
          <div className="flex-1 min-h-[300px]">
            <AttentionHeatmap matrix={result?.attention_matrix} tokens={result?.input_tokens} />
          </div>
          
          <div className="flex-1 min-h-[300px]">
            <EmbeddingVisualizer embeddings={result?.embeddings} tokens={result?.input_tokens} />
          </div>
        </div>

      </main>
    </div>
  );
}
