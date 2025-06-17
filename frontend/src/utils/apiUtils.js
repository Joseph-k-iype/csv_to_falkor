export const getApiConfig = () => {
    const currentPort = window.location.port;
    const isDevelopment = process.env.NODE_ENV === 'development';
    
    const baseURL = process.env.REACT_APP_API_URL || 
      (isDevelopment ? 'http://localhost:5000' : window.location.origin);
    
    console.log(`Frontend running on port: ${currentPort || '80/443'}`);
    console.log(`API configured for: ${baseURL}`);
    
    return {
      baseURL,
      timeout: 45000,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Frontend-Port': currentPort || 'default',
      }
    };
  };
  
  export const API_CONFIG = getApiConfig();