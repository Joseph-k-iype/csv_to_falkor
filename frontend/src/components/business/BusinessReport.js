import React from 'react';
import { 
  FileText, 
  Clock, 
  CheckCircle, 
  Zap, 
  Database, 
  Shield, 
  Network, 
  Brain,
  TrendingUp,
  Target,
  Lightbulb
} from 'lucide-react';
import { useFormatters } from '../../hooks/useFormatters';

const BusinessReport = ({ data, query, timestamp, metadata }) => {
  const { formatAdvancedInsight } = useFormatters();

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-2xl border border-white/20 shadow-xl p-6 mb-6 business-report">
      <div className="flex items-start justify-between mb-6 pb-4 border-b border-gray-200">
        <div className="flex-1">
          <h2 className="text-xl font-bold text-gray-800 mb-2 flex items-center">
            <FileText className="w-6 h-6 mr-2 text-blue-600" />
            Advanced Graph Intelligence Report
          </h2>
          <p className="text-sm text-gray-600 mb-2">Query: "{query}"</p>
          <div className="flex items-center text-xs text-gray-500 space-x-4">
            <div className="flex items-center">
              <Clock className="w-4 h-4 mr-1" />
              {new Date(timestamp).toLocaleString()}
            </div>
            {metadata?.analysis_type && (
              <div className="flex items-center">
                <Brain className="w-4 h-4 mr-1" />
                {metadata.analysis_type}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <div className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium flex items-center">
            <CheckCircle className="w-3 h-3 mr-1" />
            FalkorDB Validated
          </div>
          <div className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium flex items-center">
            <Zap className="w-3 h-3 mr-1" />
            Multi-Agent Analysis
          </div>
        </div>
      </div>

      <div className="space-y-6">
        {formatAdvancedInsight(data)}
      </div>

      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-gray-500">
          <div className="flex items-center">
            <Database className="w-3 h-3 mr-1" />
            FalkorDB Engine
          </div>
          <div className="flex items-center">
            <Shield className="w-3 h-3 mr-1" />
            Query Validation
          </div>
          <div className="flex items-center">
            <Network className="w-3 h-3 mr-1" />
            Graph Analysis
          </div>
          <div className="flex items-center">
            <Brain className="w-3 h-3 mr-1" />
            AI Reasoning
          </div>
        </div>
      </div>
    </div>
  );
};

export default React.memo(BusinessReport);