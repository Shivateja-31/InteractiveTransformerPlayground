import React from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, Cell, ZAxis } from 'recharts';

export default function EmbeddingVisualizer({ embeddings, tokens }) {
  if (!embeddings || !tokens || embeddings.length === 0) {
    return (
      <div className="bg-surface p-6 rounded-xl border border-white/5 h-64 flex items-center justify-center text-textSecondary">
        Enter text to view embeddings
      </div>
    );
  }

  // To display embeddings (32D) in 2D, we will just use the first two principal components.
  // Since we can't easily run PCA in browser without a heavy lib, we'll just take the first two dimensions
  // of the embedding vector for a naive but fast visualization.
  const data = tokens.map((t, i) => ({
    name: t.text,
    x: embeddings[i][0],
    y: embeddings[i][1],
    z: 100 // size for bubble
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border border-white/10 p-2 rounded text-xs text-white">
          <p className="font-mono mb-1">{`Token: '${payload[0].payload.name}'`}</p>
          <p className="text-textSecondary">{`Dim 1: ${payload[0].value.toFixed(3)}`}</p>
          <p className="text-textSecondary">{`Dim 2: ${payload[1].value.toFixed(3)}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-surface p-6 rounded-xl border border-white/5 flex flex-col h-full">
      <h3 className="text-sm font-semibold text-textSecondary uppercase tracking-wider mb-4">Token Embeddings (2D Slice)</h3>
      
      <div className="flex-1 w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
            <XAxis type="number" dataKey="x" name="Dim 1" stroke="#52525b" tick={{fill: '#94a3b8', fontSize: 12}} />
            <YAxis type="number" dataKey="y" name="Dim 2" stroke="#52525b" tick={{fill: '#94a3b8', fontSize: 12}} />
            <ZAxis type="number" dataKey="z" range={[100, 100]} />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="Tokens" data={data}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={`hsl(${(index * 360) / data.length}, 70%, 60%)`} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
