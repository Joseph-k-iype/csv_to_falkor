"""
Enhanced Flask Backend API for FalkorDB Graph RAG Search Engine
Supports all advanced reasoning features from the original engine
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from functools import wraps

# Import the original enhanced FalkorDB engine
try:
    from graphrag import EnhancedGraphRAGEngine, EnterpriseConfig, FalkorDBQueryValidator
except ImportError as e:
    print(f"Warning: Could not import enhanced engine: {e}")
    print("Please ensure paste.py (your original engine code) is in the same directory")
    EnhancedGraphRAGEngine = None
    EnterpriseConfig = None
    FalkorDBQueryValidator = None

app = Flask(__name__)

# Enhanced CORS configuration for multiple ports
CORS(app, 
     origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Frontend-Port"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global enhanced engine instance
engine = None
engine_config = None
system_info = {
    'initialized': False,
    'initialization_time': None,
    'total_agents': 0,
    'features': [],
    'version': '2.0.0-enhanced'
}

def timing_decorator(f):
    """Decorator to measure function execution time"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{f.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def async_timing_decorator(f):
    """Decorator for async functions to measure execution time"""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await f(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{f.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def initialize_enhanced_engine():
    """Initialize the enhanced FalkorDB engine with full capabilities"""
    global engine, engine_config, system_info
    
    if not EnhancedGraphRAGEngine:
        logger.error("Enhanced engine class not available")
        return False
    
    try:
        logger.info("Initializing Enhanced FalkorDB Graph RAG Engine...")
        
        # Load configuration
        config = EnterpriseConfig()
        config.validate()
        engine_config = config
        
        # Initialize enhanced engine with all features
        engine = EnhancedGraphRAGEngine(
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
            falkordb_host=config.falkordb_host,
            falkordb_port=config.falkordb_port,
            graph_name=config.graph_name
        )
        
        # Update system info
        system_info.update({
            'initialized': True,
            'initialization_time': datetime.now().isoformat(),
            'total_agents': 5,  # FalkorDB schema, strategy, execution, intelligence, followup
            'features': [
                'Multi-agent reasoning',
                'Query validation',
                'Indirect path exploration', 
                'Business intelligence synthesis',
                'Follow-up question generation',
                'Strategic analysis',
                'Risk assessment',
                'Operational insights'
            ],
            'database_info': {
                'host': config.falkordb_host,
                'port': config.falkordb_port,
                'graph_name': config.graph_name,
                'query_language': 'openCypher',
                'validation_enabled': True
            }
        })
        
        logger.info("Enhanced FalkorDB engine initialized successfully with full reasoning capabilities")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced engine: {e}")
        system_info.update({
            'initialized': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        return False

# Health and system endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check with system information"""
    return jsonify({
        'status': 'healthy',
        'service': 'FalkorDB Enhanced Graph RAG API',
        'version': system_info['version'],
        'engine_initialized': engine is not None,
        'system_info': system_info,
        'timestamp': datetime.now().isoformat(),
        'supported_features': {
            'multi_agent_reasoning': True,
            'query_validation': True,
            'business_intelligence': True,
            'follow_up_questions': True,
            'indirect_exploration': True,
            'strategic_analysis': True,
            'risk_assessment': True
        }
    })

@app.route('/api/connection/test', methods=['GET'])
def test_connection():
    """Enhanced connection test with comprehensive validation"""
    if not engine:
        return jsonify({
            'status': 'failed',
            'error': 'Enhanced engine not initialized',
            'system_info': system_info
        }), 500
    
    try:
        # Test connection with enhanced capabilities
        result = engine.test_connection()
        
        # Add enhanced system information
        result.update({
            'enhanced_features': system_info['features'],
            'total_agents': system_info['total_agents'],
            'multi_agent_ready': True,
            'system_version': system_info['version']
        })
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Enhanced connection test failed: {e}")
        return jsonify({
            'status': 'failed',
            'error': str(e),
            'system_info': system_info
        }), 500

@app.route('/api/graph/info', methods=['GET'])
def get_enhanced_graph_info():
    """Get comprehensive graph information with enhanced features"""
    if not engine:
        return jsonify({'error': 'Enhanced engine not initialized'}), 500
    
    try:
        # Get enhanced graph information
        info = engine.get_graph_info()
        
        # Add enhanced capabilities information
        info.update({
            'enhanced_capabilities': {
                'multi_agent_reasoning': True,
                'advanced_query_validation': True,
                'business_intelligence_synthesis': True,
                'strategic_recommendation_engine': True,
                'risk_assessment_framework': True,
                'indirect_relationship_exploration': True
            },
            'agent_information': {
                'falkordb_schema_agent': 'Schema analysis and business domain identification',
                'falkordb_strategy_agent': 'Strategic query planning and analysis approach',
                'falkordb_execution_agent': 'Query execution with validation',
                'falkordb_intelligence_agent': 'Business intelligence synthesis',
                'falkordb_followup_agent': 'Intelligent follow-up question generation'
            },
            'system_info': system_info
        })
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Enhanced graph info retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

# Basic search endpoint
@app.route('/api/search', methods=['POST'])
@timing_decorator
def basic_search():
    """Basic search with enhanced engine"""
    if not engine:
        return jsonify({'error': 'Enhanced engine not initialized'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        thread_id = data.get('thread_id', 'default')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"Processing basic search: {query[:100]}...")
        
        # Use enhanced engine for basic search
        result = engine.search_sync(query, thread_id)
        
        return jsonify({
            'result': result,
            'query': query,
            'thread_id': thread_id,
            'analysis_type': 'basic_enhanced_search',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Basic enhanced search failed: {e}")
        return jsonify({'error': str(e)}), 500

# Advanced search with full reasoning capabilities
@app.route('/api/search/followups', methods=['POST'])
@timing_decorator
def advanced_search_with_reasoning():
    """Advanced search with full multi-agent reasoning and follow-ups"""
    if not engine:
        return jsonify({'error': 'Enhanced engine not initialized'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        thread_id = data.get('thread_id', 'default')
        enable_advanced_reasoning = data.get('enable_advanced_reasoning', True)
        max_agents = data.get('max_agents', 5)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"Processing advanced search with reasoning: {query[:100]}...")
        logger.info(f"Advanced reasoning enabled: {enable_advanced_reasoning}, Max agents: {max_agents}")
        
        start_time = time.time()
        
        # Use enhanced engine with full capabilities
        result, followups = engine.search_with_followups_sync(query, thread_id)
        
        execution_time = time.time() - start_time
        
        # Enhanced response with comprehensive metadata
        response = {
            'result': result,
            'followups': followups,
            'query': query,
            'thread_id': thread_id,
            'analysis_metadata': {
                'analysis_type': 'advanced_multi_agent_reasoning',
                'agents_used': max_agents,
                'advanced_reasoning_enabled': enable_advanced_reasoning,
                'response_time': round(execution_time, 2),
                'query_validation': 'enabled',
                'business_intelligence': 'enabled',
                'strategic_analysis': 'enabled'
            },
            'system_capabilities': {
                'query_validation': True,
                'multi_agent_reasoning': True,
                'business_synthesis': True,
                'follow_up_generation': True,
                'indirect_exploration': True
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Advanced search completed in {execution_time:.2f} seconds")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Advanced search with reasoning failed: {e}")
        return jsonify({
            'error': str(e),
            'error_type': 'advanced_reasoning_failure',
            'timestamp': datetime.now().isoformat(),
            'suggestions': [
                'Check if FalkorDB is running and accessible',
                'Verify OpenAI API key is valid',
                'Try a simpler query first',
                'Check backend logs for detailed error information'
            ]
        }), 500

# Query validation endpoint
@app.route('/api/validate/query', methods=['POST'])
def validate_query():
    """Validate openCypher query for FalkorDB compatibility"""
    if not FalkorDBQueryValidator:
        return jsonify({'error': 'Query validator not available'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Validate query using enhanced validator
        validation_result = FalkorDBQueryValidator.validate_query(query)
        
        return jsonify({
            'validation_result': validation_result,
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Query validation failed: {e}")
        return jsonify({'error': str(e)}), 500

# Schema analysis endpoint
@app.route('/api/analyze/schema', methods=['GET'])
@timing_decorator
def analyze_schema():
    """Perform comprehensive schema analysis"""
    if not engine:
        return jsonify({'error': 'Enhanced engine not initialized'}), 500
    
    try:
        # Get comprehensive schema information
        schema_info = engine.get_graph_info()
        
        return jsonify({
            'schema_analysis': schema_info,
            'analysis_type': 'comprehensive_schema_analysis',
            'capabilities': system_info['features'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Schema analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

# Business intelligence endpoint
@app.route('/api/intelligence/business', methods=['POST'])
@timing_decorator
def business_intelligence_analysis():
    """Dedicated business intelligence analysis endpoint"""
    if not engine:
        return jsonify({'error': 'Enhanced engine not initialized'}), 500
    
    try:
        data = request.get_json()
        business_question = data.get('question', '')
        focus_area = data.get('focus_area', 'general')  # strategic, operational, risk, technical
        thread_id = data.get('thread_id', 'business_intelligence')
        
        if not business_question:
            return jsonify({'error': 'Business question is required'}), 400
        
        logger.info(f"Processing business intelligence analysis: {business_question[:100]}...")
        
        # Enhanced business question formatting
        enhanced_question = f"""
        Business Intelligence Analysis Request:
        
        Question: {business_question}
        Focus Area: {focus_area}
        
        Please provide a comprehensive business analysis including:
        1. Executive summary of key findings
        2. Strategic implications and recommendations
        3. Operational considerations
        4. Risk assessment
        5. Next steps and action items
        
        Use FalkorDB-compatible openCypher queries and multi-agent reasoning.
        """
        
        result, followups = engine.search_with_followups_sync(enhanced_question, thread_id)
        
        return jsonify({
            'business_analysis': result,
            'strategic_followups': followups,
            'original_question': business_question,
            'focus_area': focus_area,
            'analysis_metadata': {
                'analysis_type': 'business_intelligence',
                'multi_agent_reasoning': True,
                'business_focus': focus_area
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Business intelligence analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

# System capabilities endpoint
@app.route('/api/capabilities', methods=['GET'])
def get_system_capabilities():
    """Get comprehensive system capabilities information"""
    return jsonify({
        'system_capabilities': {
            'multi_agent_reasoning': {
                'enabled': True,
                'agents': [
                    'FalkorDB Schema Agent',
                    'FalkorDB Strategy Agent', 
                    'FalkorDB Execution Agent',
                    'FalkorDB Intelligence Agent',
                    'FalkorDB Follow-up Agent'
                ],
                'description': 'Advanced multi-agent workflow for comprehensive graph analysis'
            },
            'query_validation': {
                'enabled': True,
                'features': [
                    'openCypher compatibility checking',
                    'FalkorDB feature validation',
                    'Prevention of unsupported operations',
                    'Query optimization suggestions'
                ]
            },
            'business_intelligence': {
                'enabled': True,
                'features': [
                    'Executive summary generation',
                    'Strategic recommendation synthesis',
                    'Risk assessment framework',
                    'Operational insights analysis'
                ]
            },
            'advanced_analysis': {
                'enabled': True,
                'features': [
                    'Indirect relationship exploration',
                    'Multi-hop path analysis',
                    'Bridge entity identification',
                    'Cluster analysis',
                    'Pattern recognition'
                ]
            }
        },
        'system_info': system_info,
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/health',
            '/api/connection/test',
            '/api/graph/info',
            '/api/search',
            '/api/search/followups',
            '/api/validate/query',
            '/api/analyze/schema',
            '/api/intelligence/business',
            '/api/capabilities'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat(),
        'system_status': system_info
    }), 500

# Development endpoints (only in debug mode)
if app.debug:
    @app.route('/api/debug/system', methods=['GET'])
    def debug_system_info():
        """Debug endpoint for system information"""
        return jsonify({
            'system_info': system_info,
            'engine_status': engine is not None,
            'config_status': engine_config is not None,
            'environment_variables': {
                'OPENAI_API_KEY': 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET',
                'FALKORDB_HOST': os.getenv('FALKORDB_HOST', 'localhost'),
                'FALKORDB_PORT': os.getenv('FALKORDB_PORT', '6379'),
                'GRAPH_NAME': os.getenv('GRAPH_NAME', 'test_cor')
            }
        })

if __name__ == '__main__':
    print("🚀 Starting Enhanced FalkorDB Graph RAG API...")
    print("=" * 60)
    print(f"🔧 Version: {system_info['version']}")
    print(f"🤖 Multi-Agent Reasoning: Enabled")
    print(f"✅ Query Validation: Enabled") 
    print(f"📊 Business Intelligence: Enabled")
    print(f"🔍 Advanced Analysis: Enabled")
    print("=" * 60)
    
    if initialize_enhanced_engine():
        print("✅ Enhanced engine initialized successfully")
        print(f"🔗 Available agents: {system_info['total_agents']}")
        print(f"📋 Features: {', '.join(system_info['features'])}")
        print("🚀 Starting Flask server...")
        print("📡 CORS enabled for ports 3000 and 3001")
        print("=" * 60)
        
        # Run with enhanced configuration
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            threaded=True  # Enable threading for better performance
        )
    else:
        print("❌ Failed to initialize enhanced engine")
        print("Please check your configuration:")
        print("- OPENAI_API_KEY environment variable")
        print("- FalkorDB server running on specified host/port")
        print("- paste.py file with enhanced engine code")
        print("- All required Python dependencies installed")
        exit(1)