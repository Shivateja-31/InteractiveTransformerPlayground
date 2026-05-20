import React from 'react';
import { motion } from 'framer-motion';

export default function TokenProbabilities({ topTokens }) {
  if (!topTokens || topTokens.length === 0) {
    return (
      <div className="bg-surface p-6 rounded-xl border border-white/5 h-64 flex items-center justify-center text-textSecondary">
        Enter text to view probabilities
      </div>
    );
  }

  return (
    <div className="bg-surface p-6 rounded-xl border border-white/5 flex flex-col h-full">
      <h3 className="text-sm font-semibold text-textSecondary uppercase tracking-wider mb-4">Next Token Probability</h3>
      
      <div className="flex-1 flex flex-col justify-center space-y-4">
        {topTokens.map((t, i) => (
          <div key={i} className="flex items-center group">
            <div className="w-12 text-sm font-mono text-textPrimary truncate" title={t.token === ' ' ? 'SPACE' : t.token}>
              {t.token === ' ' ? 'SPC' : (t.token === '\\n' ? '\\n' : t.token)}
            </div>
            <div className="flex-1 h-3 bg-surfaceHover rounded-full overflow-hidden mx-3">
              <motion.div 
                className="h-full bg-accent"
                initial={{ width: 0 }}
                animate={{ width: `${t.probability * 100}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
            </div>
            <div className="w-12 text-xs text-right text-textSecondary">
              {(t.probability * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
