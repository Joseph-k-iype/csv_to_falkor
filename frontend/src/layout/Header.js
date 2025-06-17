import React from 'react';
import { Database } from 'lucide-react';
import { ConnectionStatus } from '../ui';
import { useConnectionContext } from '../../contexts/ConnectionContext';

const Header = () => {
  const { connectionStatus, testConnection } = useConnectionContext();
  
  return (
    <header className="w-full p-6">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-white/60 backdrop-blur-sm rounded-xl">
              <Database className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800">FalkorDB Intelligence</h1>
              <p className="text-sm text-gray-600">
                Advanced Graph Analytics • Port {window.location.port || '3001'}
              </p>
            </div>
          </div>
          
          <div className="hidden md:block">
            <ConnectionStatus status={connectionStatus} onRetry={testConnection} />
          </div>
        </div>
      </div>
    </header>
  );
};

export default React.memo(Header);