import React from 'react';
import { Clock, Brain, AlertCircle } from 'lucide-react';
import { BusinessReport, FollowUpQuestions } from '../business';
import { useSearchContext } from '../../contexts/SearchContext';

const ResultItem = ({ result }) => {
  const { handleFollowUpSelect, loading } = useSearchContext();
  
  return (
    <div className="space-y-4">
      <div className="bg-white/60 backdrop-blur-sm rounded-xl border border-white/30 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Clock className="w-4 h-4" />
            <span>{new Date(result.timestamp).toLocaleString()}</span>
            {result.metadata?.total_agents && (
              <React.Fragment>
                <span>•</span>
                <Brain className="w-4 h-4" />
                <span>{result.metadata.total_agents} AI Agents</span>
              </React.Fragment>
            )}
          </div>
        </div>
        <h3 className="text-lg font-medium text-gray-800 mt-2">"{result.query}"</h3>
      </div>
      
      {result.isError ? (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6">
          <div className="flex items-center space-x-2 mb-3">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <h3 className="font-medium text-red-800">System Error</h3>
          </div>
          <p className="text-red-700 whitespace-pre-wrap">{result.result}</p>
        </div>
      ) : (
        <BusinessReport 
          data={result.result}
          query={result.query}
          timestamp={result.timestamp}
          metadata={result.metadata}
        />
      )}
      
      <FollowUpQuestions
        questions={result.followups}
        onQuestionSelect={handleFollowUpSelect}
        isLoading={loading}
        metadata={result.metadata}
      />
    </div>
  );
};

const ResultsDisplay = () => {
  const { results, resultsRef } = useSearchContext();

  if (results.length === 0) return null;

  return (
    <div ref={resultsRef} className="space-y-8">
      {results.map((result) => (
        <ResultItem key={result.id} result={result} />
      ))}
    </div>
  );
};

export default ResultsDisplay;