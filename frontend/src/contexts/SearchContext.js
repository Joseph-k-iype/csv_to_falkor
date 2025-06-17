import React, { createContext, useContext, useEffect } from 'react';
import { useSearch } from '../hooks/useSearch';
import { useConnectionContext } from './ConnectionContext';

const SearchContext = createContext();

export const useSearchContext = () => {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error('useSearchContext must be used within a SearchProvider');
  }
  return context;
};

export const SearchProvider = ({ children }) => {
  const searchData = useSearch();
  const { connectionStatus } = useConnectionContext();
  
  // Initialize welcome message when connection status changes
  useEffect(() => {
    if (connectionStatus && !searchData.hasSearched) {
      searchData.initializeWelcomeMessage(connectionStatus);
    }
  }, [connectionStatus, searchData.hasSearched]);
  
  return (
    <SearchContext.Provider value={searchData}>
      {children}
    </SearchContext.Provider>
  );
};