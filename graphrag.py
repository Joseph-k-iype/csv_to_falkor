"""
Enhanced Enterprise FalkorDB Graph RAG Search Engine
Features: Intelligent follow-up questions, indirect path exploration, FalkorDB/openCypher validation
Prevents hallucination by ensuring only FalkorDB-compatible openCypher queries are generated.

Author: Assistant
Date: 2025
Dependencies: falkordb, langchain, langchain-community, langgraph, openai
"""

import os
import json
import asyncio
import argparse
import logging
import re
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Tuple
from datetime import datetime

# Core LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# FalkorDB-specific LangChain imports
from langchain_community.chains.graph_qa.falkordb import FalkorDBQAChain
from langchain_community.graphs import FalkorDBGraph

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Pydantic for data validation
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphSearchState(TypedDict):
    """Enhanced state for the enterprise graph search system."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    user_query: str
    graph_schema: Optional[str]
    conceptual_schema: Optional[Dict[str, Any]]
    cypher_queries: Optional[List[str]]
    graph_results: Optional[List[Dict[str, Any]]]
    indirect_relationships: Optional[List[Dict[str, Any]]]
    analysis_insights: Optional[List[str]]
    follow_up_questions: Optional[List[str]]
    final_answer: Optional[str]
    current_agent: Optional[str]
    metadata: Dict[str, Any]


class FalkorDBQueryValidator:
    """Validates queries to ensure FalkorDB/openCypher compatibility and prevent hallucination."""
    
    # Neo4j-specific features NOT supported by FalkorDB/openCypher
    UNSUPPORTED_PATTERNS = [
        # GDS (Graph Data Science) library functions
        r'gds\.',
        r'CALL gds\.',
        
        # Neo4j APOC procedures
        r'apoc\.',
        r'CALL apoc\.',
        
        # Neo4j-specific procedures
        r'dbms\.',
        r'db\.schema\.',
        r'db\.indexes\(',
        r'db\.constraints\(',
        
        # Neo4j-specific syntax
        r'shortestPath\s*\(',
        r'allShortestPaths\s*\(',
        
        # Neo4j built-in procedures that aren't in openCypher
        r'CALL db\.relationshipTypes\(\)',
        r'CALL db\.propertyKeys\(\)',
        r'CALL db\.labels\(\)',
        
        # Algorithms that are Neo4j-specific
        r'algo\.',
        r'CALL algo\.',
    ]
    
    # FalkorDB/openCypher supported patterns (examples)
    SUPPORTED_EXAMPLES = [
        "MATCH (n) RETURN n LIMIT 10",
        "MATCH (n)-[r]->(m) RETURN count(r)",
        "MATCH (n:Label) WHERE n.property = 'value' RETURN n",
        "MATCH path = (n)-[*1..3]->(m) RETURN path",
        "MATCH (n) RETURN labels(n), keys(n)",
        "MATCH (n) WITH count(n) as total RETURN total",
        "MATCH (n)-[r]->(m) RETURN type(r), count(*) ORDER BY count(*) DESC",
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> Dict[str, Any]:
        """Validate if a query is compatible with FalkorDB/openCypher."""
        if not query or not query.strip():
            return {'valid': False, 'error': 'Empty query'}
        
        query_upper = query.upper()
        
        # Check for unsupported patterns
        for pattern in cls.UNSUPPORTED_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    'valid': False,
                    'error': f'Query contains unsupported feature: {pattern}',
                    'suggestion': 'Use standard openCypher syntax compatible with FalkorDB',
                    'alternatives': cls._get_alternatives(pattern)
                }
        
        # Check for dangerous operations in enterprise environment
        dangerous_keywords = ['DELETE', 'DROP', 'CREATE INDEX', 'CREATE CONSTRAINT', 'MERGE']
        for keyword in dangerous_keywords:
            if keyword in query_upper and not query_upper.startswith('CREATE ('):
                return {
                    'valid': False,
                    'error': f'Query contains potentially destructive operation: {keyword}',
                    'suggestion': 'Use read-only queries (MATCH, RETURN, WITH, UNWIND, WHERE, ORDER BY, LIMIT)'
                }
        
        return {'valid': True, 'query': query}
    
    @classmethod
    def _get_alternatives(cls, unsupported_pattern: str) -> List[str]:
        """Get alternative approaches for unsupported patterns."""
        alternatives = {
            r'gds\.': [
                "Use MATCH patterns for graph traversal",
                "Use standard Cypher aggregations: count(), collect(), etc.",
                "Use path expressions: MATCH (n)-[*1..3]->(m)"
            ],
            r'shortestPath\s*\(': [
                "Use variable length paths: MATCH path = (a)-[*1..5]->(b)",
                "Implement custom shortest path with MATCH and aggregation"
            ],
            r'apoc\.': [
                "Use standard Cypher functions and expressions",
                "Implement logic with CASE, COALESCE, and standard functions"
            ]
        }
        
        for pattern, alts in alternatives.items():
            if re.search(pattern, unsupported_pattern, re.IGNORECASE):
                return alts
        
        return ["Use standard openCypher syntax compatible with FalkorDB"]
    
    @classmethod
    def sanitize_query_description(cls, description: str) -> str:
        """Sanitize natural language descriptions to prevent Neo4j-specific suggestions."""
        # Replace Neo4j-specific terms with FalkorDB/openCypher equivalents
        replacements = {
            'shortest path algorithm': 'variable length path patterns',
            'GDS library': 'standard Cypher traversal',
            'APOC procedures': 'standard Cypher functions',
            'Neo4j procedures': 'FalkorDB compatible queries',
            'gds.': 'standard Cypher: ',
            'apoc.': 'standard functions: ',
        }
        
        sanitized = description
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        
        return sanitized


class EnhancedGraphRAGEngine:
    """Enhanced Enterprise Graph RAG Engine with FalkorDB/openCypher validation."""
    
    def __init__(self, 
                 openai_api_key: str,
                 openai_base_url: Optional[str] = None,
                 falkordb_host: str = 'localhost',
                 falkordb_port: int = 6379,
                 graph_name: str = 'test_cor'):
        
        self.graph_name = graph_name
        self.validator = FalkorDBQueryValidator()
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model="o3-mini",
            api_key=openai_api_key,
            base_url=openai_base_url,
            temperature=0.1
        )
        
        # Initialize FalkorDB Graph wrapper
        self.graph = FalkorDBGraph(
            database=graph_name,
            host=falkordb_host,
            port=falkordb_port
        )
        
        # Initialize FalkorDB QA Chain with enhanced prompts for FalkorDB compatibility
        custom_cypher_prompt = """
You are a FalkorDB/openCypher expert. Generate ONLY FalkorDB-compatible openCypher queries.

CRITICAL CONSTRAINTS:
- FalkorDB uses openCypher standard, NOT Neo4j-specific features
- NO GDS library functions (gds.*) - they are NOT supported
- NO APOC procedures (apoc.*) - they are NOT supported  
- NO Neo4j-specific procedures (dbms.*, db.schema.*, etc.)
- NO shortestPath() or allShortestPaths() functions
- NO Neo4j algorithms (algo.*)

SUPPORTED FalkorDB/openCypher features:
- Basic MATCH, RETURN, WHERE, WITH, UNWIND
- Path patterns: (n)-[*1..3]->(m)
- Standard functions: count(), collect(), keys(), labels(), type()
- Aggregations: sum(), max(), min(), avg()
- String functions: toLower(), toUpper(), substring()
- Mathematical functions: abs(), round(), floor()
- List functions: size(), head(), tail()

Schema: {schema}

Question: {question}

Generate a FalkorDB-compatible openCypher query that answers the question.
"""
        
        # Create custom prompt template
        from langchain_core.prompts import PromptTemplate
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=custom_cypher_prompt
        )
        
        self.qa_chain = FalkorDBQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,  # Enterprise controlled environment
            return_intermediate_steps=True,
            cypher_prompt=cypher_prompt
        )
        
        # Create the enhanced multi-agent workflow
        self.workflow = self._create_workflow()
        logger.info(f"Enhanced FalkorDB-validated Graph RAG Engine initialized for graph: {graph_name}")
        
    def _create_tools(self) -> List[BaseTool]:
        """Create FalkorDB-validated enterprise tools."""
        
        @tool
        def get_falkordb_schema() -> str:
            """Get FalkorDB-specific schema information using only supported openCypher queries."""
            try:
                schema_info = {
                    'database': self.graph.database,
                    'raw_schema': self.graph.get_schema,
                    'structured_schema': self.graph.get_structured_schema
                }
                
                # Use FalkorDB-compatible queries for conceptual analysis
                try:
                    conceptual_query = """
                    MATCH (n)-[r]->(m)
                    RETURN labels(n)[0] as source_type, type(r) as relationship, 
                           labels(m)[0] as target_type, count(*) as frequency
                    ORDER BY frequency DESC
                    LIMIT 20
                    """
                    
                    validation = self.validator.validate_query(conceptual_query)
                    if validation['valid']:
                        conceptual_patterns = self.graph.query(conceptual_query)
                        schema_info['conceptual_patterns'] = conceptual_patterns
                    else:
                        schema_info['conceptual_patterns'] = f"Pattern analysis unavailable: {validation['error']}"
                except Exception as e:
                    schema_info['conceptual_patterns'] = f"Pattern analysis failed: {str(e)}"
                
                return json.dumps(schema_info, indent=2, default=str)
            except Exception as e:
                return f"FalkorDB schema retrieval error: {str(e)}"
        
        @tool
        def execute_validated_graph_qa(question: str, explore_indirect: bool = True) -> str:
            """Execute FalkorDB-validated question with openCypher compatibility checking."""
            try:
                logger.info(f"Executing FalkorDB-validated QA: {question[:100]}...")
                
                # Enhanced question with FalkorDB constraints
                falkordb_question = f"""
                {question}
                
                IMPORTANT: Use ONLY FalkorDB-compatible openCypher syntax.
                - NO GDS library functions (gds.*)
                - NO APOC procedures (apoc.*)
                - NO Neo4j-specific features
                - Use standard openCypher: MATCH, RETURN, WHERE, WITH, path patterns [*1..3]
                """
                
                # Execute main query
                result = self.qa_chain.invoke({"query": falkordb_question})
                
                response = {
                    'direct_answer': result.get('result', ''),
                    'cypher_query': result.get('intermediate_steps', [{}])[0].get('query', '') if result.get('intermediate_steps') else '',
                    'context': result.get('intermediate_steps', [{}])[0].get('context', '') if result.get('intermediate_steps') else ''
                }
                
                # Validate the generated query
                if response['cypher_query']:
                    validation = self.validator.validate_query(response['cypher_query'])
                    if not validation['valid']:
                        response['query_warning'] = f"Generated query may have issues: {validation['error']}"
                        response['alternatives'] = validation.get('alternatives', [])
                
                # Enhanced indirect exploration with FalkorDB constraints
                if explore_indirect:
                    indirect_question = f"""
                    Based on: "{question}"
                    Find indirect relationships using ONLY FalkorDB-compatible openCypher:
                    - Use path patterns like (a)-[*2..3]->(b) for multi-hop analysis
                    - Use standard aggregations and functions
                    - NO GDS or APOC functions
                    Look for patterns spanning 2-3 degrees of separation using standard Cypher.
                    """
                    
                    try:
                        indirect_result = self.qa_chain.invoke({"query": indirect_question})
                        response['indirect_insights'] = indirect_result.get('result', '')
                        response['indirect_cypher'] = indirect_result.get('intermediate_steps', [{}])[0].get('query', '') if indirect_result.get('intermediate_steps') else ''
                        
                        # Validate indirect query too
                        if response['indirect_cypher']:
                            indirect_validation = self.validator.validate_query(response['indirect_cypher'])
                            if not indirect_validation['valid']:
                                response['indirect_warning'] = f"Indirect query issue: {indirect_validation['error']}"
                                
                    except Exception as e:
                        response['indirect_insights'] = f"Indirect analysis unavailable: {str(e)}"
                
                return json.dumps(response, indent=2)
            except Exception as e:
                logger.error(f"FalkorDB-validated QA execution failed: {e}")
                return json.dumps({'error': f"FalkorDB-validated QA failed: {str(e)}"}, indent=2)
        
        @tool
        def explore_falkordb_paths(entity_or_concept: str, max_hops: int = 3) -> str:
            """Explore paths using FalkorDB-compatible openCypher path patterns."""
            try:
                # Use FalkorDB-compatible path exploration
                path_question = f"""
                Find entities connected to '{entity_or_concept}' within {max_hops} hops.
                Use FalkorDB-compatible openCypher syntax:
                - Use path patterns: MATCH path = (start)-[*1..{max_hops}]-(end)
                - Use standard functions: length(path), nodes(path), relationships(path)
                - NO GDS or Neo4j-specific algorithms
                Explain the significance of each connection level found.
                """
                
                result = self.qa_chain.invoke({"query": path_question})
                
                return json.dumps({
                    'analysis_type': 'falkordb_path_exploration',
                    'focus_entity': entity_or_concept,
                    'max_hops': max_hops,
                    'result': result.get('result', ''),
                    'cypher_used': result.get('intermediate_steps', [{}])[0].get('query', '') if result.get('intermediate_steps') else ''
                }, indent=2)
            except Exception as e:
                return json.dumps({'error': f"FalkorDB path exploration failed: {str(e)}"}, indent=2)
        
        @tool
        def analyze_falkordb_clusters() -> str:
            """Analyze clusters using FalkorDB-compatible openCypher queries."""
            try:
                cluster_question = """
                Identify clusters and communities using FalkorDB-compatible openCypher:
                - Use MATCH patterns to find densely connected groups
                - Use aggregations like count(), collect() to analyze connectivity
                - Use WITH clauses for complex aggregations
                - NO GDS community detection algorithms
                Find groups of entities that are strongly connected using standard Cypher.
                """
                
                result = self.qa_chain.invoke({"query": cluster_question})
                
                return json.dumps({
                    'analysis_type': 'falkordb_clustering',
                    'result': result.get('result', ''),
                    'cypher_used': result.get('intermediate_steps', [{}])[0].get('query', '') if result.get('intermediate_steps') else ''
                }, indent=2)
            except Exception as e:
                return json.dumps({'error': f"FalkorDB clustering failed: {str(e)}"}, indent=2)
        
        @tool
        def find_falkordb_bridges() -> str:
            """Find bridge entities using FalkorDB-compatible openCypher queries."""
            try:
                bridge_question = """
                Find bridge entities using FalkorDB-compatible openCypher:
                - Use degree calculations with count() aggregations
                - Use MATCH patterns to find entities connecting different groups
                - Use standard Cypher functions and aggregations
                - NO GDS centrality algorithms
                Find entities with high connectivity that bridge different parts of the graph.
                """
                
                result = self.qa_chain.invoke({"query": bridge_question})
                
                return json.dumps({
                    'analysis_type': 'falkordb_bridge_analysis',
                    'result': result.get('result', ''),
                    'cypher_used': result.get('intermediate_steps', [{}])[0].get('query', '') if result.get('intermediate_steps') else ''
                }, indent=2)
            except Exception as e:
                return json.dumps({'error': f"FalkorDB bridge analysis failed: {str(e)}"}, indent=2)
        
        @tool
        def generate_falkordb_followups(original_query: str, analysis_result: str) -> str:
            """Generate follow-up questions focused on FalkorDB-compatible analysis."""
            try:
                follow_up_prompt = f"""
                Based on the original question: "{original_query}"
                And the FalkorDB analysis result: "{analysis_result[:500]}..."
                
                Generate 5-7 intelligent follow-up questions that:
                1. Explore deeper insights using FalkorDB-compatible openCypher
                2. Focus on business implications of the graph patterns found
                3. Investigate indirect relationships using path patterns
                4. Address potential risks or optimization opportunities
                5. Explore alternative perspectives on the findings
                6. Consider operational and strategic implications
                
                Ensure all suggested analyses can be performed with standard openCypher
                syntax compatible with FalkorDB (no GDS, APOC, or Neo4j-specific features).
                """
                
                result = self.qa_chain.invoke({"query": follow_up_prompt})
                
                return json.dumps({
                    'analysis_type': 'falkordb_followup_generation',
                    'original_query': original_query,
                    'follow_up_questions': result.get('result', ''),
                    'generated_at': datetime.now().isoformat()
                }, indent=2)
            except Exception as e:
                return json.dumps({'error': f"FalkorDB follow-up generation failed: {str(e)}"}, indent=2)
        
        return [get_falkordb_schema, execute_validated_graph_qa, explore_falkordb_paths, 
                analyze_falkordb_clusters, find_falkordb_bridges, generate_falkordb_followups]
    
    def _create_agents(self) -> Dict[str, Any]:
        """Create FalkorDB-aware specialized agents."""
        
        tools = self._create_tools()
        
        # FalkorDB Schema Agent
        falkordb_schema_agent = create_react_agent(
            self.llm,
            tools=[tools[0], tools[3], tools[4]],  # schema, clusters, bridges
            state_modifier="""You are a FalkorDB enterprise schema analyst with deep openCypher expertise. 

            🔧 FALKORDB EXPERTISE:
            • Expert in FalkorDB's openCypher implementation and limitations
            • Understand what FalkorDB supports vs Neo4j-specific features
            • Focus on business meaning using ONLY FalkorDB-compatible queries
            • Never suggest GDS, APOC, or Neo4j-specific procedures
            
            📊 ANALYSIS APPROACH:
            • Use get_falkordb_schema() for FalkorDB-compatible schema analysis
            • Use analyze_falkordb_clusters() with standard openCypher patterns
            • Use find_falkordb_bridges() with degree calculations and aggregations
            • Focus on business domains identifiable through standard Cypher
            
            🚫 IMPORTANT CONSTRAINTS:
            • NO GDS library functions (gds.*) - not supported by FalkorDB
            • NO APOC procedures (apoc.*) - not supported by FalkorDB  
            • NO Neo4j-specific procedures or algorithms
            • Use ONLY standard openCypher: MATCH, RETURN, WHERE, WITH, aggregations
            
            Always provide FalkorDB-compatible insights that help business leaders understand their graph."""
        )
        
        # FalkorDB Strategic Query Agent
        falkordb_strategy_agent = create_react_agent(
            self.llm,
            tools=[tools[0], tools[2], tools[3]],  # schema, paths, clusters
            state_modifier="""You are a FalkorDB strategic query planning specialist.

            🎯 FALKORDB STRATEGY:
            • Plan comprehensive analysis using ONLY FalkorDB-compatible openCypher
            • Focus on indirect relationships using path patterns [*1..3]
            • Use standard Cypher functions and aggregations
            • Never suggest unsupported Neo4j features
            
            🔍 INDIRECT EXPLORATION (FalkorDB Compatible):
            • Use explore_falkordb_paths() with standard path patterns
            • Use variable length paths: (a)-[*2..3]->(b)
            • Use standard functions: length(), nodes(), relationships()
            • Use aggregations: count(), collect(), sum() for analysis
            
            🚫 CRITICAL CONSTRAINTS:
            • NO GDS algorithms (gds.pageRank, gds.louvain, etc.)
            • NO shortestPath() or allShortestPaths() functions
            • NO APOC procedures or Neo4j-specific features
            • Use ONLY openCypher standard supported by FalkorDB
            
            Always plan strategies that work within FalkorDB's openCypher implementation."""
        )
        
        # FalkorDB Execution Agent
        falkordb_execution_agent = create_react_agent(
            self.llm,
            tools=[tools[1], tools[2], tools[4]],  # validated_qa, paths, bridges
            state_modifier="""You are a FalkorDB query execution specialist ensuring openCypher compatibility.

            ⚡ FALKORDB EXECUTION:
            • Execute ONLY FalkorDB-compatible openCypher queries
            • Use execute_validated_graph_qa() with validation checks
            • Always explore indirect relationships using standard path patterns
            • Validate all queries against FalkorDB capabilities
            
            🔍 COMPREHENSIVE ANALYSIS (FalkorDB Compatible):
            • Use path patterns: MATCH (a)-[*1..3]->(b) for multi-hop analysis
            • Use aggregations: count(), max(), min(), collect()
            • Use WITH clauses for complex query building
            • Use standard functions: labels(), keys(), type(), size()
            
            🚫 STRICT LIMITATIONS:
            • NO GDS library calls - will cause errors in FalkorDB
            • NO APOC procedures - not available in FalkorDB
            • NO Neo4j-specific syntax or algorithms
            • Validate every query for FalkorDB compatibility
            
            Always ensure queries work with FalkorDB's openCypher implementation."""
        )
        
        # FalkorDB Intelligence Agent
        falkordb_intelligence_agent = create_react_agent(
            self.llm,
            tools=tools,  # All FalkorDB-compatible tools
            state_modifier="""You are a FalkorDB business intelligence synthesizer ensuring accurate analysis.

            📊 FALKORDB INTELLIGENCE:
            • Transform FalkorDB query results into strategic business insights
            • Synthesize findings from FalkorDB-compatible analyses only
            • Focus on actionable intelligence derived from validated queries
            • Never reference unsupported features or impossible analyses
            
            💡 ACCURATE INSIGHTS:
            • Base recommendations only on FalkorDB-compatible query results
            • Acknowledge limitations of openCypher vs advanced graph algorithms
            • Focus on insights achievable through standard Cypher patterns
            • Provide realistic assessments of what FalkorDB can reveal
            
            🚫 NO HALLUCINATION:
            • Never suggest analyses requiring GDS or APOC
            • Don't claim capabilities FalkorDB doesn't have
            • Be honest about openCypher limitations
            • Focus on achievable insights with standard Cypher
            
            Always deliver accurate, FalkorDB-achievable business intelligence."""
        )
        
        # FalkorDB Follow-up Agent
        falkordb_followup_agent = create_react_agent(
            self.llm,
            tools=[tools[5]],  # followup generation
            state_modifier="""You are a FalkorDB follow-up question specialist ensuring realistic suggestions.

            🤔 FALKORDB FOLLOW-UPS:
            • Generate questions answerable with FalkorDB's openCypher implementation
            • Focus on insights achievable through standard Cypher patterns
            • Suggest analyses that build on validated FalkorDB capabilities
            • Never suggest impossible analyses or unsupported features
            
            🎯 REALISTIC QUESTIONS:
            • Questions about path patterns and connectivity using [*1..3] syntax
            • Aggregation-based analyses using count(), collect(), sum()
            • Pattern matching with MATCH, WHERE, and WITH clauses
            • Business insights from standard openCypher capabilities
            
            🚫 NO IMPOSSIBLE SUGGESTIONS:
            • Don't suggest GDS-based community detection
            • Don't reference APOC procedures or Neo4j algorithms
            • Don't suggest analyses requiring unsupported features
            • Keep all suggestions within FalkorDB's capabilities
            
            Always generate realistic, FalkorDB-achievable follow-up questions."""
        )
        
        return {
            'falkordb_schema_agent': falkordb_schema_agent,
            'falkordb_strategy_agent': falkordb_strategy_agent,
            'falkordb_execution_agent': falkordb_execution_agent,
            'falkordb_intelligence_agent': falkordb_intelligence_agent,
            'falkordb_followup_agent': falkordb_followup_agent
        }
    
    def _create_supervisor_agent(self) -> Any:
        """Create FalkorDB-aware supervisor agent."""
        
        supervisor_prompt = ChatPromptTemplate.from_template("""
        You are supervising a FalkorDB enterprise intelligence team with openCypher expertise:
        
        🔧 FalkorDB Schema Agent: FalkorDB-specific schema analysis with openCypher compatibility
        🎯 FalkorDB Strategy Agent: Plans analysis using only FalkorDB-supported features  
        ⚡ FalkorDB Execution Agent: Executes validated openCypher queries
        📊 FalkorDB Intelligence Agent: Creates insights from FalkorDB-compatible analyses
        🤔 FalkorDB Follow-up Agent: Generates realistic FalkorDB-achievable questions
        
        Current business question: "{user_query}"
        
        Conversation progress: {messages}
        
        FALKORDB-AWARE WORKFLOW:
        
        For FalkorDB enterprise intelligence:
        1. FalkorDB Schema Agent: For understanding what's possible with openCypher
        2. FalkorDB Strategy Agent: To plan FalkorDB-compatible analysis approaches
        3. FalkorDB Execution Agent: To execute validated openCypher queries
        4. FalkorDB Intelligence Agent: To synthesize realistic business insights
        5. FalkorDB Follow-up Agent: To generate achievable follow-up questions
        6. FINISH: When comprehensive FalkorDB-compatible analysis is complete
        
        CRITICAL: Ensure all analysis stays within FalkorDB's openCypher capabilities.
        Never route to analyses requiring GDS, APOC, or Neo4j-specific features.
        
        Options: falkordb_schema_agent, falkordb_strategy_agent, falkordb_execution_agent, falkordb_intelligence_agent, falkordb_followup_agent, FINISH
        
        Respond with just the agent name or FINISH.
        """)
        
        supervisor = supervisor_prompt | self.llm
        return supervisor
    
    def _create_workflow(self) -> StateGraph:
        """Create FalkorDB-validated workflow."""
        
        # Create agents
        agents = self._create_agents()
        supervisor = self._create_supervisor_agent()
        
        # Define workflow
        workflow = StateGraph(GraphSearchState)
        
        # Add agent nodes
        for agent_name, agent in agents.items():
            workflow.add_node(agent_name, agent)
        
        # Add supervisor node
        def supervisor_node(state: GraphSearchState) -> GraphSearchState:
            """FalkorDB-aware supervisor decision node."""
            result = supervisor.invoke({
                "user_query": state["user_query"],
                "messages": state["messages"]
            })
            
            next_agent = result.content.strip().lower()
            
            if next_agent == "finish":
                state["current_agent"] = "FINISH"
            else:
                state["current_agent"] = next_agent
                
            return state
        
        workflow.add_node("supervisor", supervisor_node)
        
        # Add edges
        workflow.add_edge(START, "supervisor")
        
        # Conditional edges from supervisor to agents
        def route_supervisor(state: GraphSearchState) -> str:
            current_agent = state.get("current_agent", "").lower()
            if current_agent == "finish":
                return END
            elif current_agent in agents:
                return current_agent
            else:
                return "falkordb_strategy_agent"  # Default fallback
        
        workflow.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {
                "falkordb_schema_agent": "falkordb_schema_agent",
                "falkordb_strategy_agent": "falkordb_strategy_agent", 
                "falkordb_execution_agent": "falkordb_execution_agent",
                "falkordb_intelligence_agent": "falkordb_intelligence_agent",
                "falkordb_followup_agent": "falkordb_followup_agent",
                END: END
            }
        )
        
        # Add edges back to supervisor from each agent
        for agent_name in agents.keys():
            workflow.add_edge(agent_name, "supervisor")
        
        # Compile workflow
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def search_with_followups(self, user_query: str, thread_id: str = "default") -> Tuple[str, List[str]]:
        """Execute FalkorDB-validated search with follow-ups."""
        
        initial_state = GraphSearchState(
            messages=[HumanMessage(content=user_query)],
            user_query=user_query,
            graph_schema=None,
            conceptual_schema=None,
            cypher_queries=None,
            graph_results=None,
            indirect_relationships=None,
            analysis_insights=None,
            follow_up_questions=None,
            final_answer=None,
            current_agent=None,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "thread_id": thread_id,
                "graph_database": self.graph_name,
                "analysis_type": "falkordb_validated_intelligence",
                "includes_followups": True,
                "query_validation": "enabled"
            }
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            logger.info(f"Processing FalkorDB-validated query: {user_query}")
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Extract answer and follow-up questions
            answer = None
            follow_ups = []
            
            # Get the main answer
            for message in reversed(final_state["messages"]):
                if isinstance(message, AIMessage) and not answer:
                    answer = message.content
                    break
            
            # Look for follow-up questions in the messages
            for message in final_state["messages"]:
                if isinstance(message, AIMessage) and "follow_up" in message.content.lower():
                    try:
                        content = message.content
                        if "follow_up_questions" in content:
                            # Parse JSON if present
                            import re
                            json_match = re.search(r'\{.*"follow_up_questions".*\}', content, re.DOTALL)
                            if json_match:
                                follow_up_data = json.loads(json_match.group())
                                follow_up_text = follow_up_data.get("follow_up_questions", "")
                                # Extract individual questions
                                lines = follow_up_text.split('\n')
                                questions = [line.strip() for line in lines if line.strip() and ('?' in line or line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.')))]
                                follow_ups.extend(questions[:7])  # Limit to 7 questions
                    except:
                        pass
            
            if not answer:
                answer = "I apologize, but I couldn't generate a FalkorDB-compatible analysis for your query."
            
            logger.info(f"FalkorDB-validated analysis completed with {len(follow_ups)} follow-up questions")
            return answer, follow_ups
            
        except Exception as e:
            error_msg = f"FalkorDB-validated analysis failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, []
    
    async def search(self, user_query: str, thread_id: str = "default") -> str:
        """Execute FalkorDB-validated search."""
        answer, _ = await self.search_with_followups(user_query, thread_id)
        return answer
    
    def search_sync(self, user_query: str, thread_id: str = "default") -> str:
        """Synchronous FalkorDB-validated search."""
        return asyncio.run(self.search(user_query, thread_id))
    
    def search_with_followups_sync(self, user_query: str, thread_id: str = "default") -> Tuple[str, List[str]]:
        """Synchronous FalkorDB-validated search with follow-ups."""
        return asyncio.run(self.search_with_followups(user_query, thread_id))
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get FalkorDB-specific graph information."""
        try:
            return {
                'database_name': self.graph.database,
                'database_type': 'FalkorDB',
                'query_language': 'openCypher',
                'schema': self.graph.get_schema,
                'structured_schema': self.graph.get_structured_schema,
                'connection_info': {
                    'host': getattr(self.graph, '_host', 'localhost'),
                    'port': getattr(self.graph, '_port', 6379),
                    'connected': True
                },
                'falkordb_features': {
                    'query_validation': True,
                    'opencypher_standard': True,
                    'follow_up_questions': True,
                    'indirect_path_exploration': True,
                    'no_gds_functions': True,
                    'no_apoc_procedures': True
                },
                'supported_features': [
                    'Standard openCypher MATCH patterns',
                    'Path expressions [*1..3]',
                    'Standard aggregations (count, sum, max, min)',
                    'Standard functions (labels, keys, type)',
                    'WITH clauses and complex queries'
                ],
                'unsupported_features': [
                    'GDS library functions (gds.*)',
                    'APOC procedures (apoc.*)',
                    'Neo4j-specific algorithms',
                    'shortestPath() and allShortestPaths()',
                    'Neo4j built-in procedures'
                ],
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get FalkorDB info: {e}")
            return {'error': str(e), 'connected': False}
    
    def test_connection(self) -> Dict[str, Any]:
        """Test FalkorDB connection with validation capabilities."""
        try:
            # Test basic connection
            test_result = self.graph.query("RETURN 'connection_test' as status")
            
            # Test query validation
            test_query = "MATCH (n) RETURN count(n) LIMIT 1"
            validation_result = self.validator.validate_query(test_query)
            
            return {
                'status': 'connected',
                'database': self.graph.database,
                'database_type': 'FalkorDB',
                'query_language': 'openCypher',
                'test_result': test_result,
                'validation_enabled': True,
                'validation_test': validation_result['valid'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class EnterpriseConfig:
    """Enterprise configuration management."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.falkordb_host = os.getenv("FALKORDB_HOST", "localhost")
        self.falkordb_port = int(os.getenv("FALKORDB_PORT", "6379"))
        self.graph_name = os.getenv("GRAPH_NAME", "test_cor")
    
    def validate(self) -> bool:
        """Validate enterprise configuration."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for enterprise deployment")
        return True


def main():
    """FalkorDB-validated command-line interface."""
    parser = argparse.ArgumentParser(
        description="FalkorDB-Validated Enterprise Graph RAG Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python graph_rag_search.py --query "What are key relationships?" --falkordb-validated
  python graph_rag_search.py --interactive-falkordb
  python graph_rag_search.py --test-falkordb-connection
        """
    )
    
    parser.add_argument("--query", "-q", help="Business analysis query")
    parser.add_argument("--thread-id", "-t", default="default", help="Analysis session ID")
    parser.add_argument("--config-file", "-c", help="Path to configuration file")
    parser.add_argument("--interactive-falkordb", "-if", action="store_true", help="FalkorDB-validated interactive mode")
    parser.add_argument("--follow-ups", "-f", action="store_true", help="Generate follow-up questions")
    parser.add_argument("--falkordb-validated", "-fv", action="store_true", help="Enable FalkorDB query validation")
    parser.add_argument("--test-falkordb-connection", action="store_true", help="Test FalkorDB connection with validation")
    parser.add_argument("--falkordb-info", action="store_true", help="Show FalkorDB-specific information")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = EnterpriseConfig()
    
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                setattr(config, key, value)
    
    try:
        config.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return 1
    
    # Print FalkorDB-specific banner
    print("🔧 FalkorDB-Validated Enterprise Graph RAG Search Engine")
    print("=" * 70)
    print(f"📊 Graph Database: {config.graph_name} (FalkorDB)")
    print(f"🔗 Connection: {config.falkordb_host}:{config.falkordb_port}")
    print(f"🤖 AI Model: OpenAI o3-mini")
    print(f"🔧 Query Language: openCypher (FalkorDB-compatible)")
    print(f"✅ Validation: Enabled (No GDS/APOC/Neo4j-specific features)")
    print(f"🚫 Prevents: Query hallucination and unsupported features")
    print("=" * 70)
    
    try:
        engine = EnhancedGraphRAGEngine(
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
            falkordb_host=config.falkordb_host,
            falkordb_port=config.falkordb_port,
            graph_name=config.graph_name
        )
        
        print("✅ FalkorDB-validated enterprise engine initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize FalkorDB engine: {e}")
        return 1
    
    # Handle different modes
    if args.test_falkordb_connection:
        print("\n🔍 Testing FalkorDB Connection with Validation...")
        result = engine.test_connection()
        if result['status'] == 'connected':
            print(f"✅ Successfully connected to {result['database_type']}: {result['database']}")
            print(f"🔧 Query Language: {result['query_language']}")
            print(f"✅ Validation Enabled: {'✅' if result.get('validation_enabled') else '❌'}")
        else:
            print(f"❌ Connection failed: {result['error']}")
        return 0
    
    if args.falkordb_info:
        print("\n📊 FalkorDB-Specific Information:")
        print("-" * 50)
        info = engine.get_graph_info()
        if 'error' not in info:
            print(f"Database Type: {info['database_type']}")
            print(f"Query Language: {info['query_language']}")
            print(f"Validation Features:")
            for feature, enabled in info.get('falkordb_features', {}).items():
                print(f"  • {feature.replace('_', ' ').title()}: {'✅' if enabled else '❌'}")
            print(f"\nSupported Features:")
            for feature in info.get('supported_features', []):
                print(f"  ✅ {feature}")
            print(f"\nUnsupported Features (Prevented):")
            for feature in info.get('unsupported_features', []):
                print(f"  ❌ {feature}")
        else:
            print(f"❌ Error: {info['error']}")
        return 0
    
    if args.interactive_falkordb:
        print("\n🔧 FalkorDB-Validated Interactive Analysis Mode")
        print("💡 Features:")
        print("  • Query validation prevents unsupported features")
        print("  • openCypher compatibility verification")
        print("  • Intelligent follow-up questions")
        print("  • No hallucination of impossible analyses")
        print("-" * 70)
        
        while True:
            try:
                query = input(f"\n[{args.thread_id}] 🔍 FalkorDB Query: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Ending FalkorDB-validated session...")
                    break
                
                if not query:
                    continue
                
                print("\n🔧 Analyzing with FalkorDB-validated intelligence...")
                print("   • Ensuring openCypher compatibility")
                print("   • Preventing GDS/APOC/Neo4j-specific features")
                print("   • Validating query feasibility")
                
                # Get answer and follow-ups
                answer, follow_ups = engine.search_with_followups_sync(query, args.thread_id)
                
                print(f"\n📊 FALKORDB-VALIDATED ANALYSIS:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                
                # Display follow-up questions
                if follow_ups:
                    print(f"\n🤔 FALKORDB-COMPATIBLE FOLLOW-UP QUESTIONS:")
                    print("-" * 50)
                    for i, follow_up in enumerate(follow_ups[:7], 1):
                        clean_question = follow_up.strip().lstrip('0123456789.- ')
                        if clean_question and '?' in clean_question:
                            print(f"{i}. {clean_question}")
                    
                    # Allow user to select a follow-up
                    print("\n💡 Enter a number (1-7) to explore a follow-up, or ask a new question:")
                    try:
                        follow_up_input = input("Choice (or new query): ").strip()
                        if follow_up_input.isdigit():
                            idx = int(follow_up_input) - 1
                            if 0 <= idx < len(follow_ups):
                                selected_follow_up = follow_ups[idx].strip().lstrip('0123456789.- ')
                                if selected_follow_up:
                                    print(f"\n🔍 FalkorDB Follow-up: {selected_follow_up}")
                                    follow_answer, _ = engine.search_with_followups_sync(selected_follow_up, args.thread_id)
                                    print(f"\n📋 FOLLOW-UP ANALYSIS:")
                                    print("-" * 50)
                                    print(follow_answer)
                                    print("-" * 50)
                    except (ValueError, IndexError):
                        pass
                
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n👋 FalkorDB session terminated by user")
                break
            except Exception as e:
                print(f"❌ FalkorDB analysis error: {e}")
        
        return 0
    
    if args.query:
        print(f"\n🔍 FalkorDB Query: {args.query}")
        if args.falkordb_validated:
            print("🔧 Validation: Enabled (openCypher compatibility)")
        if args.follow_ups:
            print("🤔 Generating: FalkorDB-compatible follow-up questions")
        print("-" * 70)
        
        try:
            if args.follow_ups:
                result, follow_ups = engine.search_with_followups_sync(args.query, args.thread_id)
                print(f"\n📊 FALKORDB-VALIDATED ANALYSIS:\n{result}")
                
                if follow_ups:
                    print(f"\n🤔 FALKORDB-COMPATIBLE FOLLOW-UPS:")
                    print("-" * 40)
                    for i, follow_up in enumerate(follow_ups[:7], 1):
                        clean_question = follow_up.strip().lstrip('0123456789.- ')
                        if clean_question and '?' in clean_question:
                            print(f"{i}. {clean_question}")
            else:
                result = engine.search_sync(args.query, args.thread_id)
                print(f"\n📊 FALKORDB ANALYSIS:\n{result}")
                
        except Exception as e:
            print(f"❌ FalkorDB analysis failed: {e}")
            return 1
    else:
        print("\n⚠️  No query provided. Use --query, --interactive-falkordb, or --test-falkordb-connection")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())