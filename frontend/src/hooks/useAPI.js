import { useState, useCallback } from 'react';
import { API_CONFIG } from '../utils/apiUtils';

export const useAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiCall = useCallback(async (endpoint, method = 'GET', data = null) => {
    setLoading(true);
    setError(null);
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);
      
      const config = {
        method,
        headers: API_CONFIG.headers,
        signal: controller.signal,
      };
      
      if (data) {
        config.body = JSON.stringify(data);
      }
      
      console.log(`API Call: ${method} ${API_CONFIG.baseURL}/api${endpoint}`);
      
      const response = await fetch(`${API_CONFIG.baseURL}/api${endpoint}`, config);
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
      }
      
      const result = await response.json();
      console.log('API Response:', result);
      return result;
    } catch (error) {
      if (error.name === 'AbortError') {
        const timeoutError = new Error('Request timeout - complex analysis may take longer. Please try a simpler query first.');
        setError(timeoutError);
        throw timeoutError;
      }
      console.error('API Error:', error);
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, []);

  return { apiCall, loading, error };
};