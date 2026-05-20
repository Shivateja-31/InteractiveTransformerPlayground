import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

export default function AttentionHeatmap({ matrix, tokens }) {
  if (!matrix || !tokens || matrix.length === 0) {
    return (
      <div className="bg-surface p-6 rounded-xl border border-white/5 h-64 flex items-center justify-center text-textSecondary">
        Enter text to view attention map
      </div>
    );
  }

  // matrix is [seq_len, seq_len]
  // tokens is array of {id, char}
  const seqLen = tokens.length;

  return (
    <div className="bg-surface p-6 rounded-xl border border-white/5 flex flex-col h-full overflow-hidden">
      <h3 className="text-sm font-semibold text-textSecondary uppercase tracking-wider mb-4">Self-Attention Heatmap</h3>
      
      <div className="flex-1 overflow-auto flex items-center justify-center">
        <div 
          className="grid gap-1" 
          style={{ 
            gridTemplateColumns: `auto repeat(${seqLen}, minmax(1rem, 2rem))`,
            gridTemplateRows: `auto repeat(${seqLen}, minmax(1rem, 2rem))`
          }}
        >
          {/* Top Header */}
          <div className="w-6 h-6"></div> {/* Empty corner */}
          {tokens.map((t, i) => (
            <div key={`col-${i}`} className="text-xs text-textSecondary text-center h-6 flex flex-col justify-end truncate" title={t.text}>
              {t.text}
            </div>
          ))}

          {/* Rows */}
          {tokens.map((t, rowIdx) => (
            <React.Fragment key={`row-wrap-${rowIdx}`}>
              {/* Row Header */}
              <div className="text-xs text-textSecondary w-6 flex items-center justify-end pr-2 truncate" title={t.text}>
                {t.text}
              </div>
              
              {/* Heatmap Cells */}
              {tokens.map((_, colIdx) => {
                const val = matrix[rowIdx][colIdx];
                const bgOpacity = Math.max(0, Math.min(1, val));
                const isMasked = val === 0 && rowIdx < colIdx; // Causal mask
                
                return (
                  <motion.div 
                    key={`cell-${rowIdx}-${colIdx}`}
                    className={`w-full h-full min-w-[1rem] min-h-[1rem] sm:min-w-[1.5rem] sm:min-h-[1.5rem] rounded-sm relative group`}
                    style={{ 
                      backgroundColor: isMasked ? 'transparent' : `rgba(59, 130, 246, ${bgOpacity})`,
                      border: isMasked ? '1px dashed rgba(255,255,255,0.05)' : 'none'
                    }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: (rowIdx + colIdx) * 0.01 }}
                  >
                    <div className="absolute hidden group-hover:block z-10 bottom-full left-1/2 -translate-x-1/2 mb-1 bg-black text-white text-[10px] px-1 py-0.5 rounded pointer-events-none whitespace-nowrap">
                      {(val * 100).toFixed(1)}%
                    </div>
                  </motion.div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}
