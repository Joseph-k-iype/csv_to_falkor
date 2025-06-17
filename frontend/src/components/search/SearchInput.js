import React, { useCallback } from 'react';
import { Search, Loader2, Brain, Zap } from 'lucide-react';

const SearchInput = ({ 
  query, 
  onQueryChange, 
  onSearch, 
  isLoading, 
  placeholder = "Ask complex questions about your graph data..." 
}) => {
  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSearch();
    }
  }, [onSearch]);

  return (
    <div className="relative group">
      <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-2xl blur-lg opacity-20 group-hover:opacity-30 transition-opacity"></div>
      <div className="relative bg-white/80 backdrop-blur-xl rounded-2xl border border-white/30 shadow-xl hover:shadow-2xl transition-all duration-300">
        <div className="flex items-center p-4">
          <Search className="w-6 h-6 text-gray-400 mr-4 flex-shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            className="flex-1 text-lg bg-transparent border-none outline-none text-gray-800 placeholder-gray-500"
            disabled={isLoading}
          />
          {isLoading && (
            <div className="flex items-center mr-2">
              <Loader2 className="w-5 h-5 text-blue-500 animate-spin mr-2" />
              <span className="text-sm text-gray-500">Reasoning...</span>
            </div>
          )}
          <button
            onClick={onSearch}
            disabled={!query.trim() || isLoading}
            className="ml-2 px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 disabled:from-gray-400 disabled:to-gray-400 text-white rounded-xl font-medium transition-all duration-200 disabled:cursor-not-allowed flex items-center"
          >
            {isLoading ? (
              <React.Fragment>
                <Brain className="w-4 h-4 mr-1" />
                Analyzing...
              </React.Fragment>
            ) : (
              <React.Fragment>
                <Zap className="w-4 h-4 mr-1" />
                Analyze
              </React.Fragment>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default React.memo(SearchInput);