import React, { createContext, useContext } from 'react';
import { useConnection } from '../hooks/useConnection';

const ConnectionContext = createContext();

export const useConnectionContext = () => {
  const context = useContext(ConnectionContext);
  if (!context) {
    throw new Error('useConnectionContext must be used within a ConnectionProvider');
  }
  return context;
};

export const ConnectionProvider = ({ children }) => {
  const connectionData = useConnection();
  
  return (
    <ConnectionContext.Provider value={connectionData}>
      {children}
    </ConnectionContext.Provider>
  );
};