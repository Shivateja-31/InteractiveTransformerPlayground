import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown } from 'lucide-react';

const stages = [
  { id: 'text', label: 'Input Text' },
  { id: 'tokens', label: 'Tokenization' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'pos_encoding', label: 'Positional Encoding' },
  { id: 'attention', label: 'Self-Attention' },
  { id: 'ffn', label: 'Feed Forward Network' },
  { id: 'softmax', label: 'Softmax' },
  { id: 'output', label: 'Next Token' },
];

export default function PipelineDiagram({ activeStage = 'attention' }) {
  const activeIndex = stages.findIndex(s => s.id === activeStage);

  return (
    <div className="bg-surface p-6 rounded-xl border border-white/5 flex flex-col items-center overflow-y-auto h-full space-y-2">
      <h3 className="text-sm font-semibold text-textSecondary uppercase tracking-wider mb-4 self-start">Transformer Pipeline</h3>
      
      {stages.map((stage, i) => {
        const isActive = i === activeIndex;
        const isPast = i < activeIndex;
        
        return (
          <React.Fragment key={stage.id}>
            <motion.div 
              className={`w-full max-w-[200px] text-center p-3 rounded-lg border text-sm font-medium transition-colors ${
                isActive 
                  ? 'bg-primary/20 border-primary text-primary shadow-[0_0_15px_rgba(59,130,246,0.3)]' 
                  : isPast 
                    ? 'bg-surfaceHover border-white/10 text-textPrimary' 
                    : 'bg-transparent border-white/5 text-textSecondary'
              }`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              {stage.label}
            </motion.div>
            
            {i < stages.length - 1 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: (i + 0.5) * 0.1 }}
              >
                <ArrowDown className={`w-5 h-5 ${isPast || isActive ? 'text-primary/50' : 'text-white/10'}`} />
              </motion.div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
