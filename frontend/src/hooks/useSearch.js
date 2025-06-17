import { useState, useCallback, useRef } from 'react';
import { useAPI } from './useAPI';

export const useSearch = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);
  const { apiCall, loading } = useAPI();
  const resultsRef = useRef(null);

  const performAdvancedSearch = useCallback(async (searchQuery = query) => {
    if (!searchQuery.trim() || loading) return;
    
    setHasSearched(true);
    const searchId = `search_${Date.now()}`;
    
    try {
      console.log('Starting advanced search with full reasoning...');
      
      const response = await apiCall('/search/followups', 'POST', {
        query: searchQuery,
        thread_id: 'default',
        enable_advanced_reasoning: true,
        max_agents: 5
      });
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      const newResult = {
        id: searchId,
        query: searchQuery,
        result: response.result || 'No analysis results returned',
        followups: response.followups || [],
        timestamp: new Date().toISOString(),
        metadata: {
          analysis_type: 'advanced_multi_agent_reasoning',
          total_agents: 5,
          response_time: response.response_time,
          cypher_queries: response.cypher_queries,
          indirect_relationships: response.indirect_relationships
        }
      };
      
      setResults(prev => [newResult, ...prev]);
      setQuery('');
      
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
      
    } catch (error) {
      console.error('Advanced search failed:', error);
      const errorResult = {
        id: searchId,
        query: searchQuery,
        result: `❌ Analysis Failed: ${error.message}

This could be due to:
• Backend connection issues (check if Flask is running on port 5000)
• FalkorDB connection problems (ensure it's running on port 6379)
• Complex query timeout (try a simpler question first)
• API configuration mismatch

Please check the browser console for detailed error information.`,
        followups: [],
        timestamp: new Date().toISOString(),
        isError: true,
        metadata: {
          error_type: 'system_error',
          error_details: error.message
        }
      };
      
      setResults(prev => [errorResult, ...prev]);
    }
  }, [query, loading, apiCall]);

  const handleFollowUpSelect = useCallback((question) => {
    setQuery(question);
    performAdvancedSearch(question);
  }, [performAdvancedSearch]);

  const initializeWelcomeMessage = useCallback((connectionStatus) => {
    const welcomeMessage = `🚀 FalkorDB Intelligence System ${connectionStatus?.status === 'connected' ? 'Online' : 'Offline'}

${connectionStatus?.status === 'connected' ? 
  '✅ Multi-agent reasoning system is active and ready for complex graph analysis.\n\n🔧 Available Features:\n• Advanced query validation\n• Multi-hop relationship exploration\n• Business intelligence synthesis\n• Strategic recommendation engine\n• Risk assessment capabilities\n\n💡 Try asking complex questions about your graph data to see the full reasoning capabilities in action.' :
  '❌ Unable to connect to the FalkorDB backend.\n\nPlease ensure:\n• Flask backend is running on port 5000\n• FalkorDB is running on port 6379\n• CORS is properly configured\n• Environment variables are set correctly'
}

What would you like to explore in your graph data?`;

    setResults([{
      id: 'welcome',
      query: 'System Status',
      result: welcomeMessage,
      followups: connectionStatus?.status === 'connected' ? [
        "What are the most critical business relationships in my graph?",
        "Analyze network vulnerabilities and risk patterns",
        "Show me indirect connections that could impact operations",
        "Identify key influencers and bridge entities in the network"
      ] : [],
      timestamp: new Date().toISOString(),
      metadata: {
        system_status: connectionStatus?.status,
        analysis_type: 'system_initialization',
        total_agents: connectionStatus?.status === 'connected' ? 5 : 0
      }
    }]);
  }, []);

  return {
    query,
    setQuery,
    results,
    hasSearched,
    loading,
    resultsRef,
    performAdvancedSearch,
    handleFollowUpSelect,
    initializeWelcomeMessage
  };
};