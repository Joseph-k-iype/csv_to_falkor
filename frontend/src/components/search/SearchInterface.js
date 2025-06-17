import React from 'react';
import SearchInput from './SearchInput';
import { ConnectionStatus } from '../ui';
import { useSearchContext } from '../../contexts/SearchContext';
import { useConnectionContext } from '../../contexts/ConnectionContext';

const SearchInterface = () => {
  const { 
    query, 
    setQuery, 
    hasSearched, 
    loading, 
    performAdvancedSearch 
  } = useSearchContext();
  
  const { connectionStatus, testConnection } = useConnectionContext();

  return (
    <div className={`transition-all duration-700 ${hasSearched ? 'mb-8' : 'mt-20 mb-32'}`}>
      {!hasSearched && (
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-light text-gray-800 mb-4">
            Advanced Graph Intelligence
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Experience multi-agent AI reasoning for complex graph analysis. Ask sophisticated questions and get comprehensive business insights.
          </p>
        </div>
      )}
      
      <SearchInput
        query={query}
        onQueryChange={setQuery}
        onSearch={performAdvancedSearch}
        isLoading={loading}
      />
      
      <div className="md:hidden mt-4">
        <ConnectionStatus status={connectionStatus} onRetry={testConnection} />
      </div>
    </div>
  );
};

export default SearchInterface;