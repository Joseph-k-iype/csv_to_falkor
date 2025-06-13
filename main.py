#!/usr/bin/env python3
"""
Enhanced FalkorDB CSV to Knowledge Graph Converter
A streamlined tool for creating rich knowledge graphs from CSV data with API-based vector embeddings.

Features:
- Vector embedding support (OpenAI, UAE via API, custom endpoints)
- Advanced data quality validation
- Flexible embedding configuration
- Knowledge graph enrichment
- Performance optimizations
- Comprehensive analytics and profiling

Usage:
    python enhanced_graph_converter.py <graph_name> <csv_directory_path> <config_file_path>
"""

import os
import sys
import json
import time
import logging
import argparse
import asyncio
import aiohttp
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import openai
try:
    from falkordb import FalkorDB
    import redis
except ImportError:
    print("Error: FalkorDB and redis packages not found. Install with: pip install falkordb redis")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_graph_converter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and endpoints."""
    provider: str  # 'openai', 'uae', 'custom'
    model_name: str
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    dimensions: Optional[int] = None
    batch_size: int = 100
    max_tokens: int = 8192
    similarity_function: str = 'cosine'  # 'cosine' or 'euclidean'
    enabled: bool = True

@dataclass
class EntityLinkingConfig:
    """Configuration for entity linking and disambiguation."""
    enabled: bool = True
    confidence_threshold: float = 0.8
    max_candidates: int = 5
    use_fuzzy_matching: bool = True
    external_knowledge_base: Optional[str] = None

@dataclass
class DataQualityConfig:
    """Configuration for data quality validation."""
    enabled: bool = True
    max_null_percentage: float = 0.5
    duplicate_detection: bool = True
    outlier_detection: bool = True
    data_profiling: bool = True

class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        raise NotImplementedError

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider supporting text-embedding-3-large and text-embedding-3-small."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        if not config.api_key:
            config.api_key = os.getenv('OPENAI_API_KEY')
        if not config.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = openai.OpenAI(api_key=config.api_key)
        
        # Set default dimensions based on model
        if not config.dimensions:
            if 'text-embedding-3-large' in config.model_name:
                config.dimensions = 3072
            elif 'text-embedding-3-small' in config.model_name:
                config.dimensions = 1536
            else:
                config.dimensions = 1536
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.config.model_name,
                    dimensions=self.config.dimensions
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {e}")
                # Return zero vectors for failed batches
                embeddings.extend([[0.0] * self.config.dimensions] * len(batch))
        
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.config.model_name,
                dimensions=self.config.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            return [0.0] * self.config.dimensions

class UAEEmbeddingProvider(EmbeddingProvider):
    """UAE (Universal Angle Embedding) provider via API endpoint."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        if not config.endpoint_url:
            # Default UAE API endpoint (you can customize this)
            config.endpoint_url = "https://api.uae-embeddings.com/embed"
        
        # Set default dimensions for UAE models
        if not config.dimensions:
            if 'UAE-Large-V1' in config.model_name:
                config.dimensions = 1024
            else:
                config.dimensions = 768
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using UAE API endpoint."""
        async with aiohttp.ClientSession() as session:
            embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                payload = {
                    'texts': batch,
                    'model': self.config.model_name
                }
                
                headers = {'Content-Type': 'application/json'}
                if self.config.api_key:
                    headers['Authorization'] = f'Bearer {self.config.api_key}'
                
                try:
                    async with session.post(
                        self.config.endpoint_url,
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            batch_embeddings = result.get('embeddings', [])
                            embeddings.extend(batch_embeddings)
                        else:
                            logger.error(f"UAE API error: {response.status}")
                            embeddings.extend([[0.0] * self.config.dimensions] * len(batch))
                            
                except Exception as e:
                    logger.error(f"Error calling UAE API: {e}")
                    embeddings.extend([[0.0] * self.config.dimensions] * len(batch))
            
            return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using UAE API."""
        import requests
        
        payload = {
            'texts': [text],
            'model': self.config.model_name
        }
        
        headers = {'Content-Type': 'application/json'}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        try:
            response = requests.post(
                self.config.endpoint_url,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('embeddings', [[0.0] * self.config.dimensions])[0]
            else:
                logger.error(f"UAE API error: {response.status_code}")
                return [0.0] * self.config.dimensions
                
        except Exception as e:
            logger.error(f"Error calling UAE API: {e}")
            return [0.0] * self.config.dimensions

class CustomEmbeddingProvider(EmbeddingProvider):
    """Custom embedding provider for external APIs."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        if not config.endpoint_url:
            raise ValueError("Custom embedding provider requires endpoint_url")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using custom API endpoint."""
        async with aiohttp.ClientSession() as session:
            embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                payload = {
                    'texts': batch,
                    'model': self.config.model_name
                }
                
                headers = {}
                if self.config.api_key:
                    headers['Authorization'] = f'Bearer {self.config.api_key}'
                
                try:
                    async with session.post(
                        self.config.endpoint_url,
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            batch_embeddings = result.get('embeddings', [])
                            embeddings.extend(batch_embeddings)
                        else:
                            logger.error(f"Custom API error: {response.status}")
                            embeddings.extend([[0.0] * self.config.dimensions] * len(batch))
                            
                except Exception as e:
                    logger.error(f"Error calling custom embedding API: {e}")
                    embeddings.extend([[0.0] * self.config.dimensions] * len(batch))
            
            return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using custom API."""
        import requests
        
        payload = {
            'texts': [text],
            'model': self.config.model_name
        }
        
        headers = {}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        try:
            response = requests.post(
                self.config.endpoint_url,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('embeddings', [[0.0] * self.config.dimensions])[0]
            else:
                logger.error(f"Custom API error: {response.status_code}")
                return [0.0] * self.config.dimensions
                
        except Exception as e:
            logger.error(f"Error calling custom embedding API: {e}")
            return [0.0] * self.config.dimensions

class DataQualityAnalyzer:
    """Analyzes and validates data quality."""
    
    def __init__(self, config: DataQualityConfig):
        self.config = config
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """Analyze data quality of a DataFrame."""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_percentages': {},
            'duplicate_rows': 0,
            'data_types': {},
            'outliers': {},
            'quality_score': 0.0
        }
        
        if not self.config.enabled:
            return report
        
        # Calculate null percentages
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            report['null_percentages'][col] = null_pct
        
        # Check for duplicates
        if self.config.duplicate_detection:
            report['duplicate_rows'] = df.duplicated().sum()
        
        # Data type analysis
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)
        
        # Outlier detection for numeric columns
        if self.config.outlier_detection:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                report['outliers'][col] = len(outliers)
        
        # Calculate overall quality score
        avg_null_pct = np.mean(list(report['null_percentages'].values()))
        duplicate_pct = report['duplicate_rows'] / len(df) if len(df) > 0 else 0
        
        quality_score = max(0, 1.0 - avg_null_pct - duplicate_pct)
        report['quality_score'] = quality_score
        
        return report

class EnhancedGraphConverter:
    """Enhanced GraphConverter with vector embeddings and knowledge graph features."""
    
    def __init__(self, graph_name: str, csv_path: str, config_path: str):
        """Initialize the Enhanced GraphConverter."""
        self.graph_name = graph_name
        self.csv_path = Path(csv_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.db = None
        self.graph = None
        self.redis_client = None
        self.embedding_provider = None
        self.data_quality_analyzer = DataQualityAnalyzer(
            DataQualityConfig(**self.config.get('data_quality', {}))
        )
        
        # Statistics tracking
        self.stats = {
            'nodes_created': 0,
            'relationships_created': 0,
            'embeddings_created': 0,
            'vector_indexes_created': 0,
            'processing_time': 0,
            'files_processed': 0,
            'data_quality_reports': {},
            'errors': []
        }
        
        self._setup_database()
        self._setup_embedding_provider()
    
    def _load_config(self) -> Dict:
        """Load and validate configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Add default configurations if not present
            if 'embedding' not in config:
                config['embedding'] = {
                    'enabled': False,
                    'provider': 'openai',
                    'model_name': 'text-embedding-3-small'
                }
            
            if 'data_quality' not in config:
                config['data_quality'] = {
                    'enabled': True,
                    'validation': True,
                    'profiling': True
                }
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _setup_database(self):
        """Initialize FalkorDB connection."""
        try:
            db_config = self.config.get('database', {})
            
            connection_params = {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 6379)
            }
            
            if db_config.get('password'):
                connection_params['password'] = db_config.get('password')
            
            self.db = FalkorDB(**connection_params)
            self.graph = self.db.select_graph(self.graph_name)
            self.redis_client = redis.Redis(**connection_params, decode_responses=True)
            
            logger.info(f"Connected to FalkorDB: {self.graph_name}")
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _setup_embedding_provider(self):
        """Initialize embedding provider based on configuration."""
        embedding_config = self.config.get('embedding', {})
        
        if not embedding_config.get('enabled', False):
            logger.info("Embedding support disabled")
            return
        
        try:
            provider = embedding_config.get('provider', 'openai').lower()
            config = EmbeddingConfig(**embedding_config)
            
            if provider == 'openai':
                self.embedding_provider = OpenAIEmbeddingProvider(config)
            elif provider == 'uae':
                self.embedding_provider = UAEEmbeddingProvider(config)
            elif provider == 'custom':
                self.embedding_provider = CustomEmbeddingProvider(config)
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
            
            logger.info(f"Initialized {provider} embedding provider with model: {config.model_name}")
            
        except Exception as e:
            logger.error(f"Error setting up embedding provider: {e}")
            self.embedding_provider = None
    
    def _create_vector_indexes(self):
        """Create vector indexes for embeddings."""
        if not self.embedding_provider:
            return
        
        vector_fields = self.config.get('embedding', {}).get('vector_fields', [])
        
        for field_config in vector_fields:
            try:
                entity_pattern = field_config['entity_pattern']
                attribute = field_config['attribute']
                
                options = {
                    'dimension': self.embedding_provider.config.dimensions,
                    'similarityFunction': self.embedding_provider.config.similarity_function,
                    'M': field_config.get('M', 16),
                    'efConstruction': field_config.get('efConstruction', 200),
                    'efRuntime': field_config.get('efRuntime', 10)
                }
                
                query = f"""
                CREATE VECTOR INDEX FOR {entity_pattern} ON ({attribute}) 
                OPTIONS {json.dumps(options).replace('"', "'")}
                """
                
                self.graph.query(query)
                self.stats['vector_indexes_created'] += 1
                logger.info(f"Created vector index: {entity_pattern} ON {attribute}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if "already exists" in error_msg:
                    logger.debug(f"Vector index already exists: {entity_pattern}")
                else:
                    logger.error(f"Error creating vector index: {e}")
    
    async def _generate_embeddings_for_text_fields(self, df: pd.DataFrame, text_fields: List[str]) -> Dict[str, List[List[float]]]:
        """Generate embeddings for specified text fields."""
        if not self.embedding_provider:
            return {}
        
        embeddings = {}
        
        for field in text_fields:
            if field not in df.columns:
                continue
            
            texts = df[field].fillna('').astype(str).tolist()
            
            try:
                field_embeddings = await self.embedding_provider.embed_texts(texts)
                embeddings[field] = field_embeddings
                logger.info(f"Generated {len(field_embeddings)} embeddings for field: {field}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for field {field}: {e}")
                embeddings[field] = [[0.0] * self.embedding_provider.config.dimensions] * len(texts)
        
        return embeddings
    
    def _process_enhanced_nodes(self, file_config: Dict) -> int:
        """Process nodes with enhanced features including embeddings and semantic analysis."""
        csv_file = self.csv_path / file_config['file']
        
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return 0
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Processing {len(df)} rows from {csv_file}")
            
            # Data quality analysis
            quality_report = self.data_quality_analyzer.analyze_dataframe(df)
            self.stats['data_quality_reports'][file_config['file']] = quality_report
            logger.info(f"Data quality score for {csv_file}: {quality_report['quality_score']:.2f}")
            
            # Generate embeddings if configured
            embeddings = {}
            embedding_fields = file_config.get('embedding_fields', [])
            if embedding_fields and self.embedding_provider:
                embeddings = asyncio.run(
                    self._generate_embeddings_for_text_fields(df, embedding_fields)
                )
            
            node_label = file_config['node_label']
            field_mappings = file_config['field_mappings']
            batch_size = file_config.get('batch_size', 1000)
            
            nodes_created = 0
            
            # Process in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                
                create_statements = []
                for idx, (_, row) in enumerate(batch.iterrows()):
                    actual_idx = i + idx
                    
                    # Build basic properties
                    properties = self._build_cypher_properties(row, field_mappings)
                    
                    # Add embeddings as vector properties
                    for field, field_embeddings in embeddings.items():
                        if actual_idx < len(field_embeddings):
                            embedding = field_embeddings[actual_idx]
                            vector_field_name = f"{field}_embedding"
                            vector_str = f"vecf32({json.dumps(embedding)})"
                            properties = properties[:-1] + f", {vector_field_name}: {vector_str}" + "}"
                    
                    create_statements.append(f"CREATE (n{nodes_created}:{node_label} {properties})")
                    nodes_created += 1
                
                # Execute batch
                if create_statements:
                    query = " ".join(create_statements)
                    try:
                        self.graph.query(query)
                        logger.debug(f"Batch created {len(create_statements)} {node_label} nodes")
                    except Exception as e:
                        logger.error(f"Error creating batch of {node_label} nodes: {e}")
                        self.stats['errors'].append(f"Node creation error in {csv_file}: {e}")
            
            # Update embedding stats
            total_embeddings = sum(len(emb) for emb in embeddings.values())
            self.stats['embeddings_created'] += total_embeddings
            
            logger.info(f"Created {nodes_created} {node_label} nodes with {total_embeddings} embeddings")
            return nodes_created
            
        except Exception as e:
            logger.error(f"Error processing enhanced nodes from {csv_file}: {e}")
            self.stats['errors'].append(f"Enhanced node processing error {csv_file}: {e}")
            return 0
    
    def _build_cypher_properties(self, row: pd.Series, field_mappings: Dict, embeddings: Dict = None) -> str:
        """Build Cypher property string with optional embeddings."""
        properties = []
        
        for csv_field, graph_field in field_mappings.items():
            if csv_field in row.index:
                value = self._sanitize_value(row[csv_field])
                if value is not None:
                    if isinstance(value, str):
                        properties.append(f"{graph_field}: '{value}'")
                    else:
                        properties.append(f"{graph_field}: {value}")
        
        return "{" + ", ".join(properties) + "}"
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize values for Cypher queries."""
        if pd.isna(value) or value is None:
            return None
        
        if isinstance(value, str):
            # Enhanced string sanitization
            value = value.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            return value
        
        if isinstance(value, (int, float)):
            return value
        
        if isinstance(value, bool):
            return str(value).lower()
        
        return str(value).replace("'", "\\'").replace('"', '\\"')
    
    def _create_enhanced_indexes_and_constraints(self):
        """Create indexes, constraints, and vector indexes."""
        try:
            # Create traditional indexes
            indexes = self.config.get('indexes', [])
            for index_config in indexes:
                label = index_config['label']
                properties = index_config['properties']
                
                for prop in properties:
                    try:
                        query = f"CREATE INDEX FOR (n:{label}) ON (n.{prop})"
                        self.graph.query(query)
                        self.stats['indexes_created'] += 1
                        logger.info(f"Created index on {label}.{prop}")
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Index creation failed for {label}.{prop}: {e}")
            
            # Create constraints
            constraints = self.config.get('constraints', [])
            for constraint_config in constraints:
                label = constraint_config['label']
                property_name = constraint_config['property']
                constraint_type = constraint_config.get('type', 'UNIQUE')
                
                try:
                    if constraint_type == 'UNIQUE':
                        result = self.redis_client.execute_command(
                            'GRAPH.CONSTRAINT', 'CREATE', 
                            self.graph_name, 
                            'UNIQUE', 'NODE', label, 
                            'PROPERTIES', '1', property_name
                        )
                        self.stats['constraints_created'] += 1
                        logger.info(f"Created unique constraint on {label}.{property_name}")
                        
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Constraint creation failed: {e}")
            
            # Create vector indexes
            self._create_vector_indexes()
            
        except Exception as e:
            logger.error(f"Error creating indexes/constraints: {e}")
    
    def _generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        try:
            # Basic graph statistics
            node_count_query = "MATCH (n) RETURN count(n) as node_count"
            relationship_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
            
            node_result = self.graph.query(node_count_query)
            rel_result = self.graph.query(relationship_count_query)
            
            total_nodes = node_result.result_set[0][0] if node_result.result_set else 0
            total_relationships = rel_result.result_set[0][0] if rel_result.result_set else 0
            
            # Node and relationship distributions
            node_types_query = "MATCH (n) RETURN labels(n) as label, count(n) as count ORDER BY count DESC"
            node_types_result = self.graph.query(node_types_query)
            
            node_distribution = {}
            for row in node_types_result.result_set:
                label = row[0][0] if row[0] else 'Unknown'
                count = row[1]
                node_distribution[label] = count
            
            rel_types_query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC"
            rel_types_result = self.graph.query(rel_types_query)
            
            rel_distribution = {}
            for row in rel_types_result.result_set:
                rel_type = row[0]
                count = row[1]
                rel_distribution[rel_type] = count
            
            # Enhanced analytics
            graph_density = total_relationships / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
            
            # Vector similarity analysis (if embeddings exist)
            vector_analysis = {}
            if self.embedding_provider:
                vector_analysis = {
                    'total_embeddings': self.stats['embeddings_created'],
                    'vector_dimensions': self.embedding_provider.config.dimensions,
                    'similarity_function': self.embedding_provider.config.similarity_function,
                    'vector_indexes': self.stats['vector_indexes_created']
                }
            
            report = {
                'summary': {
                    'total_nodes': total_nodes,
                    'total_relationships': total_relationships,
                    'graph_density': graph_density,
                    'processing_time_seconds': self.stats['processing_time'],
                    'files_processed': self.stats['files_processed']
                },
                'distributions': {
                    'nodes': node_distribution,
                    'relationships': rel_distribution
                },
                'embeddings': vector_analysis,
                'data_quality': self.stats['data_quality_reports'],
                'performance_stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def convert(self):
        """Main enhanced conversion process."""
        start_time = time.time()
        
        try:
            logger.info("Starting enhanced CSV to FalkorDB knowledge graph conversion...")
            
            # Create enhanced indexes and constraints
            self._create_enhanced_indexes_and_constraints()
            
            # Process node files with enhanced features
            node_files = self.config.get('node_files', [])
            for file_config in node_files:
                nodes_created = self._process_enhanced_nodes(file_config)
                self.stats['nodes_created'] += nodes_created
                self.stats['files_processed'] += 1
            
            # Process relationship files (using existing logic)
            relationship_files = self.config.get('relationship_files', [])
            for file_config in relationship_files:
                relationships_created = self._process_relationships(file_config)
                self.stats['relationships_created'] += relationships_created
                self.stats['files_processed'] += 1
            
            # Calculate processing time
            self.stats['processing_time'] = time.time() - start_time
            
            # Generate comprehensive report
            comprehensive_report = self._generate_comprehensive_report()
            
            # Save enhanced report
            report_path = f"{self.graph_name}_enhanced_report.json"
            with open(report_path, 'w') as f:
                json.dump(comprehensive_report, f, indent=2)
            
            logger.info("Enhanced conversion completed successfully!")
            logger.info(f"Comprehensive report saved to: {report_path}")
            logger.info(f"Total nodes created: {self.stats['nodes_created']}")
            logger.info(f"Total relationships created: {self.stats['relationships_created']}")
            logger.info(f"Total embeddings created: {self.stats['embeddings_created']}")
            logger.info(f"Vector indexes created: {self.stats['vector_indexes_created']}")
            logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
            
            if self.stats['errors']:
                logger.warning(f"Encountered {len(self.stats['errors'])} errors during processing")
            
        except Exception as e:
            logger.error(f"Enhanced conversion failed: {e}")
            raise
        finally:
            if self.db:
                self.db.close()
            if self.redis_client:
                self.redis_client.close()
    
    def _process_relationships(self, file_config: Dict) -> int:
        """Process relationships (keeping original logic for compatibility)."""
        csv_file = self.csv_path / file_config['file']
        
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return 0
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Processing {len(df)} relationships from {csv_file}")
            
            relationships_created = 0
            batch_size = file_config.get('batch_size', 1000)
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                
                for idx, row in batch.iterrows():
                    rel_config = file_config['relationship']
                    
                    source_key = row[rel_config['source']['csv_field']]
                    target_key = row[rel_config['target']['csv_field']]
                    
                    if pd.isna(source_key) or pd.isna(target_key):
                        continue
                    
                    source_match = self._build_match_clause(row, rel_config['source'], "source")
                    target_match = self._build_match_clause(row, rel_config['target'], "target")
                    
                    rel_properties = ""
                    if 'properties' in rel_config:
                        props = self._build_cypher_properties(row, rel_config['properties'])
                        if props != "{}":
                            rel_properties = f" {props}"
                    
                    rel_type = rel_config['type']
                    query = f"""
                    MATCH {source_match}
                    MATCH {target_match}
                    MERGE (source)-[r:{rel_type}{rel_properties}]->(target)
                    """
                    
                    try:
                        self.graph.query(query)
                        relationships_created += 1
                    except Exception as e:
                        logger.warning(f"Error creating relationship: {e}")
                        self.stats['errors'].append(f"Relationship creation error: {e}")
            
            logger.info(f"Created {relationships_created} relationships from {csv_file}")
            return relationships_created
            
        except Exception as e:
            logger.error(f"Error processing relationships from {csv_file}: {e}")
            self.stats['errors'].append(f"Relationship processing error {csv_file}: {e}")
            return 0
    
    def _build_match_clause(self, row: pd.Series, node_config: Dict, var_name: str = "n") -> str:
        """Build MATCH clause for relationships."""
        label = node_config['label']
        key_field = node_config['key_field']
        csv_field = node_config['csv_field']
        
        key_value = self._sanitize_value(row[csv_field])
        
        if isinstance(key_value, str):
            return f"({var_name}:{label} {{{key_field}: '{key_value}'}})"
        else:
            return f"({var_name}:{label} {{{key_field}: {key_value}}})"


def main():
    """Main entry point for the enhanced converter."""
    parser = argparse.ArgumentParser(
        description="Enhanced CSV to FalkorDB Knowledge Graph Converter with API-based Vector Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
- Vector embeddings (OpenAI, UAE via API, custom endpoints)
- Data quality analysis and validation
- Performance optimizations
- Comprehensive analytics

Examples:
    python enhanced_graph_converter.py ecommerce_graph ./csv_files ./enhanced_config.json
    python enhanced_graph_converter.py social_network /path/to/csvs /path/to/config.json
        """
    )
    
    parser.add_argument('graph_name', help='Name of the graph to create in FalkorDB')
    parser.add_argument('csv_path', help='Path to directory containing CSV files')
    parser.add_argument('config_path', help='Path to enhanced configuration JSON file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV directory not found: {args.csv_path}")
        sys.exit(1)
    
    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file not found: {args.config_path}")
        sys.exit(1)
    
    try:
        converter = EnhancedGraphConverter(args.graph_name, args.csv_path, args.config_path)
        converter.convert()
        print(f"Successfully created enhanced knowledge graph: {args.graph_name}")
        
    except Exception as e:
        print(f"Enhanced conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()