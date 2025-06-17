export const QUERY_TIMEOUT = 45000;
export const MAX_AGENTS = 5;
export const DEFAULT_THREAD_ID = 'default';

export const ANALYSIS_TYPES = {
  SYSTEM_INITIALIZATION: 'system_initialization',
  ADVANCED_MULTI_AGENT: 'advanced_multi_agent_reasoning',
  SYSTEM_ERROR: 'system_error'
};

export const CONNECTION_STATES = {
  CONNECTED: 'connected',
  FAILED: 'failed',
  INITIALIZING: 'initializing'
};

export const SAMPLE_QUERIES = [
  "Perform a comprehensive risk assessment of our network topology",
  "Analyze customer journey patterns and identify optimization opportunities",
  "Find hidden connections that could impact our strategic initiatives",
  "Evaluate the business impact of removing key relationship nodes"
];

export const WELCOME_FOLLOWUPS = [
  "What are the most critical business relationships in my graph?",
  "Analyze network vulnerabilities and risk patterns",
  "Show me indirect connections that could impact operations",
  "Identify key influencers and bridge entities in the network"
];

// Component size configurations
export const COMPONENT_SIZES = {
  SEARCH_INPUT: {
    sm: 'text-sm px-3 py-2',
    md: 'text-lg px-4 py-3',
    lg: 'text-xl px-6 py-4'
  },
  BUTTON: {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-6 py-2 text-base',
    lg: 'px-8 py-3 text-lg'
  }
};

// Animation durations
export const ANIMATION_DURATIONS = {
  FAST: 200,
  NORMAL: 300,
  SLOW: 500,
  VERY_SLOW: 700
};

// Breakpoints for responsive design
export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536
};
