import React from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';

const ConnectionStatus = ({ status, onRetry }) => {
  const getStatusInfo = () => {
    if (!status) {
      return {
        color: 'gray',
        icon: AlertCircle,
        text: 'Initializing connection...',
        description: 'Setting up FalkorDB multi-agent system'
      };
    }
    
    if (status.status === 'connected') {
      return {
        color: 'green',
        icon: CheckCircle,
        text: 'FalkorDB Intelligence Active',
        description: `${status.database || 'Unknown'} • Multi-agent reasoning enabled`
      };
    }
    
    return {
      color: 'red',
      icon: AlertCircle,
      text: 'Connection failed',
      description: status.error || 'Unable to connect to FalkorDB backend'
    };
  };

  const { color, icon: Icon, text, description } = getStatusInfo();
  
  return (
    <div className={`flex items-center space-x-3 p-3 rounded-lg border transition-all duration-200 ${
      color === 'green' ? 'bg-green-50 border-green-200' :
      color === 'red' ? 'bg-red-50 border-red-200' :
      'bg-gray-50 border-gray-200'
    }`}>
      <Icon className={`w-5 h-5 ${
        color === 'green' ? 'text-green-600' :
        color === 'red' ? 'text-red-600' :
        'text-gray-600'
      }`} />
      <div className="flex-1">
        <div className={`text-sm font-medium ${
          color === 'green' ? 'text-green-800' :
          color === 'red' ? 'text-red-800' :
          'text-gray-800'
        }`}>
          {text}
        </div>
        <div className="text-xs text-gray-600">{description}</div>
      </div>
      {color === 'red' && onRetry && (
        <button
          onClick={onRetry}
          className="text-xs text-red-600 hover:text-red-800 underline transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  );
};

export default React.memo(ConnectionStatus);