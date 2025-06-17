import React from 'react';
import { CheckCircle, Brain, Network } from 'lucide-react';

const Footer = () => (
  <footer className="w-full mt-16 pb-8">
    <div className="max-w-4xl mx-auto px-6">
      <div className="text-center text-sm text-gray-500 space-y-2">
        <p>FalkorDB Intelligence • Multi-Agent Reasoning • Port {window.location.port || '3001'}</p>
        <div className="flex items-center justify-center space-x-6">
          <span className="flex items-center">
            <CheckCircle className="w-4 h-4 mr-1 text-green-500" />
            Query Validation
          </span>
          <span className="flex items-center">
            <Brain className="w-4 h-4 mr-1 text-blue-500" />
            AI Reasoning
          </span>
          <span className="flex items-center">
            <Network className="w-4 h-4 mr-1 text-purple-500" />
            Graph Intelligence
          </span>
        </div>
      </div>
    </div>
  </footer>
);

export default React.memo(Footer);