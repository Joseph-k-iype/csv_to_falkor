import { useState, useEffect } from 'react';
import { useAPI } from './useAPI';

export const useConnection = () => {
  const [connectionStatus, setConnectionStatus] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const { apiCall } = useAPI();

  const testConnection = async () => {
    try {
      const response = await apiCall('/connection/test');
      setConnectionStatus(response);
      return response;
    } catch (error) {
      const errorStatus = {
        status: 'failed',
        error: error.message,
        timestamp: new Date().toISOString()
      };
      setConnectionStatus(errorStatus);
      return errorStatus;
    }
  };

  const initializeSystem = async () => {
    console.log('Initializing FalkorDB Intelligence System...');
    try {
      const [connectionResult, healthResult] = await Promise.all([
        apiCall('/connection/test').catch(err => ({ status: 'failed', error: err.message })),
        apiCall('/health').catch(err => ({ status: 'failed', error: err.message }))
      ]);
      
      setConnectionStatus(connectionResult);
      setSystemInfo(healthResult);
      
      if (connectionResult.status === 'connected') {
        try {
          const graphInfo = await apiCall('/graph/info');
          setSystemInfo(prev => ({ ...prev, graphInfo }));
        } catch (error) {
          console.warn('Could not fetch graph info:', error.message);
        }
      }
      
      return { connectionResult, healthResult };
    } catch (error) {
      console.error('System initialization failed:', error);
      setConnectionStatus({
        status: 'failed',
        error: error.message,
        timestamp: new Date().toISOString()
      });
      throw error;
    }
  };

  useEffect(() => {
    initializeSystem();
  }, []);

  return {
    connectionStatus,
    systemInfo,
    testConnection,
    initializeSystem
  };
};