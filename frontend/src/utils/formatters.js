export const formatTimestamp = (timestamp) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch (error) {
      return 'Invalid date';
    }
  };
  
  /**
   * Format port number for display
   */
  export const formatPort = () => {
    return window.location.port || '3001';
  };
  
  /**
   * Clean and format question text
   */
  export const cleanQuestionText = (question) => {
    if (!question || typeof question !== 'string') return '';
    return question.replace(/^\d+\.\s*/, '').trim();
  };
  
  /**
   * Truncate text to specified length
   */
  export const truncateText = (text, maxLength = 100) => {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };
  
  /**
   * Format error message for display
   */
  export const formatErrorMessage = (error) => {
    if (typeof error === 'string') return error;
    if (error?.message) return error.message;
    return 'An unknown error occurred';
  };
  
  /**
   * Format connection status for display
   */
  export const formatConnectionStatus = (status) => {
    if (!status) return 'Unknown';
    return status.charAt(0).toUpperCase() + status.slice(1);
  };
  
  /**
   * Generate unique ID for components
   */
  export const generateId = (prefix = 'id') => {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };
  
  /**
   * Validate URL format
   */
  export const isValidUrl = (string) => {
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;
    }
  };
  
  /**
   * Get environment-specific configuration
   */
  export const getEnvironmentConfig = () => {
    return {
      isDevelopment: process.env.NODE_ENV === 'development',
      isProduction: process.env.NODE_ENV === 'production',
      apiUrl: process.env.REACT_APP_API_URL,
      debugMode: process.env.REACT_APP_DEBUG === 'true'
    };
  };