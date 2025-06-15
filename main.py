#!/usr/bin/env python3
"""
FalkorDB CSV to Knowledge Graph Converter with OpenAI Embeddings and Persistent Caching
A streamlined tool for creating knowledge graphs from CSV data with optimized OpenAI vector embeddings.

Usage:
    python graph_converter.py <graph_name> <csv_directory_path> <config_file_path>
"""

import os
import sys
import json
import time
import logging
import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np
from openai import OpenAI

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
        logging.FileHandler('graph_converter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_to_json_serializable(obj):
    """Convert numpy types and other non-JSON serializable types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and other non-serializable objects."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

@dataclass
class DataQualityConfig:
    """Configuration for data quality validation."""
    enabled: bool = True
    max_null_percentage: float = 0.5
    duplicate_detection: bool = True
    outlier_detection: bool = True
    data_profiling: bool = True

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
        
        try:
            # Calculate null percentages
            for col in df.columns:
                null_pct = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
                report['null_percentages'][col] = convert_to_json_serializable(null_pct)
            
            # Check for duplicates
            if self.config.duplicate_detection:
                report['duplicate_rows'] = convert_to_json_serializable(df.duplicated().sum())
            
            # Data type analysis
            for col in df.columns:
                report['data_types'][col] = str(df[col].dtype)
            
            # Outlier detection for numeric columns
            if self.config.outlier_detection:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if len(df[col].dropna()) > 0:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                            report['outliers'][col] = convert_to_json_serializable(len(outliers))
                        else:
                            report['outliers'][col] = 0
            
            # Calculate overall quality score
            avg_null_pct = np.mean(list(report['null_percentages'].values())) if report['null_percentages'] else 0
            duplicate_pct = report['duplicate_rows'] / len(df) if len(df) > 0 else 0
            
            quality_score = max(0, 1.0 - avg_null_pct - duplicate_pct)
            report['quality_score'] = convert_to_json_serializable(quality_score)
            
        except Exception as e:
            logger.error(f"Error analyzing dataframe: {e}")
            report['error'] = str(e)
        
        # Convert entire report to ensure JSON serialization
        return convert_to_json_serializable(report)

class OpenAIEmbeddingProvider:
    """OpenAI embedding provider supporting text-embedding-3-large and text-embedding-3-small."""
    
    def __init__(self, model_name: str, api_key: str, dimensions: Optional[int] = None, batch_size: int = 100):
        self.model_name = model_name
        self.batch_size = batch_size
        
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Set default dimensions based on model
        if dimensions:
            self.dimensions = dimensions
        elif 'text-embedding-3-large' in model_name:
            self.dimensions = 3072
        elif 'text-embedding-3-small' in model_name:
            self.dimensions = 1536
        else:
            self.dimensions = 1536
        
        logger.info(f"Using {self.dimensions} dimensions for embeddings")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API."""
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Generating embeddings for {len(texts)} texts in {total_batches} batches")
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = texts[i:i + self.batch_size]
            
            try:
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                    dimensions=self.dimensions
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"Successfully processed batch {batch_num}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings for batch {batch_num}: {e}")
                # Return zero vectors for failed batches
                embeddings.extend([[0.0] * self.dimensions] * len(batch))
        
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name,
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            return [0.0] * self.dimensions

class GraphConverter:
    """Main class for converting CSV data to FalkorDB graph database with OpenAI embeddings."""
    
    def __init__(self, graph_name: str, csv_path: str, config_path: str):
        """Initialize the GraphConverter with configuration."""
        logger.info("Initializing GraphConverter...")
        
        self.graph_name = graph_name
        self.csv_path = Path(csv_path)
        self.config_path = Path(config_path)
        
        logger.info(f"Graph name: {graph_name}")
        logger.info(f"CSV path: {csv_path}")
        logger.info(f"Config path: {config_path}")
        
        # Load configuration
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
            'embeddings_from_cache': 0,
            'vector_indexes_created': 0,
            'indexes_created': 0,
            'constraints_created': 0,
            'processing_time': 0,
            'files_processed': 0,
            'duplicate_rows_removed': 0,
            'unique_constraint_violations_avoided': 0,
            'data_quality_reports': {},
            'errors': []
        }
        
        # Setup database and embedding provider
        self._setup_database()
        self._setup_embedding_provider()
        
        logger.info("GraphConverter initialization completed")
    
    def _load_config(self) -> Dict:
        """Load and validate configuration."""
        logger.info("Loading configuration...")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Add default configurations if not present
            if 'embedding' not in config:
                config['embedding'] = {
                    'enabled': False,
                    'model_name': 'text-embedding-3-small'
                }
            
            if 'data_quality' not in config:
                config['data_quality'] = {
                    'enabled': True,
                    'max_null_percentage': 0.5,
                    'duplicate_detection': True,
                    'outlier_detection': True,
                    'data_profiling': True
                }
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            logger.info(f"Embedding enabled: {config.get('embedding', {}).get('enabled', False)}")
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _detect_database_constraints(self):
        """Detect what constraints actually exist in the database."""
        detected_constraints = []
        try:
            constraints = self.redis_client.execute_command('GRAPH.CONSTRAINT', 'LIST', self.graph_name)
            logger.info(f"Database has {len(constraints)} existing constraints:")
            
            for constraint in constraints:
                logger.info(f"  - {constraint}")
                # Parse constraint to extract label and property
                parts = constraint.split()
                if len(parts) >= 6 and parts[0] == 'UNIQUE' and parts[1] == 'NODE':
                    label = parts[2]
                    prop_count = int(parts[4])
                    properties = parts[5:5+prop_count]
                    for prop in properties:
                        detected_constraints.append({
                            'label': label,
                            'property': prop,
                            'type': 'UNIQUE'
                        })
                        logger.info(f"    Parsed: {label}.{prop} (UNIQUE)")
            
            return detected_constraints
            
        except Exception as e:
            logger.warning(f"Could not detect database constraints: {e}")
            return []

    def _setup_database(self):
        """Initialize FalkorDB connection."""
        logger.info("Setting up database connection...")
        
        try:
            db_config = self.config.get('database', {})
            
            connection_params = {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 6379)
            }
            
            if db_config.get('password'):
                connection_params['password'] = db_config.get('password')
            
            logger.info(f"Connecting to FalkorDB at {connection_params['host']}:{connection_params['port']}")
            
            self.db = FalkorDB(**connection_params)
            self.graph = self.db.select_graph(self.graph_name)
            self.redis_client = redis.Redis(**connection_params, decode_responses=True)
            
            # Test connection
            test_result = self.graph.query("RETURN 1")
            logger.info(f"Database connection successful - Graph: {self.graph_name}")
            
            # Detect existing constraints in database
            logger.info("Checking for existing database constraints...")
            database_constraints = self._detect_database_constraints()
            config_constraints = self.config.get('constraints', [])
            
            # Check for mismatches between database and config
            if database_constraints and not config_constraints:
                logger.warning("⚠️  CONSTRAINT MISMATCH DETECTED!")
                logger.warning("The database has constraints but your config doesn't specify them.")
                logger.warning("This will cause 'unique constraint violation' errors.")
                logger.warning("")
                logger.warning("Add these to your config 'constraints' section:")
                for db_constraint in database_constraints:
                    logger.warning(f'  {{"label": "{db_constraint["label"]}", "property": "{db_constraint["property"]}", "type": "UNIQUE"}}')
                logger.warning("")
                logger.warning("Or remove constraints from database with:")
                for db_constraint in database_constraints:
                    logger.warning(f'  GRAPH.CONSTRAINT DROP {self.graph_name} UNIQUE NODE {db_constraint["label"]} PROPERTIES 1 {db_constraint["property"]}')
                logger.warning("")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.error("Make sure FalkorDB is running on the specified host and port")
            raise
    
    def _setup_embedding_provider(self):
        """Initialize OpenAI embedding provider."""
        embedding_config = self.config.get('embedding', {})
        
        if not embedding_config.get('enabled', False):
            logger.info("Embedding support disabled")
            return
        
        logger.info("Setting up OpenAI embedding provider...")
        
        try:
            api_key = embedding_config.get('api_key')
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                raise ValueError("OpenAI API key not provided in config or environment variable OPENAI_API_KEY")
            
            # Mask API key for logging
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            logger.info(f"Using API key: {masked_key}")
            
            self.embedding_provider = OpenAIEmbeddingProvider(
                model_name=embedding_config.get('model_name', 'text-embedding-3-small'),
                api_key=api_key,
                dimensions=embedding_config.get('dimensions'),
                batch_size=embedding_config.get('batch_size', 100)
            )
            
            logger.info("OpenAI embedding provider setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up OpenAI embedding provider: {e}")
            self.embedding_provider = None
    
    def _create_indexes_and_constraints(self):
        """Create indexes, constraints, and vector indexes."""
        logger.info("Creating indexes and constraints...")
        
        try:
            # Create traditional indexes
            indexes = self.config.get('indexes', [])
            logger.info(f"Creating {len(indexes)} index groups...")
            
            for index_config in indexes:
                label = index_config['label']
                properties = index_config['properties']
                
                for prop in properties:
                    try:
                        query = f"CREATE INDEX FOR (n:{label}) ON (n.{prop})"
                        self.graph.query(query)
                        self.stats['indexes_created'] += 1
                        logger.debug(f"Created index on {label}.{prop}")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "already indexed" in error_msg or "already exists" in error_msg:
                            logger.debug(f"Index already exists for {label}.{prop}")
                        else:
                            logger.warning(f"Index creation failed for {label}.{prop}: {e}")
            
            # Create constraints
            constraints = self.config.get('constraints', [])
            logger.info(f"Creating {len(constraints)} constraints...")
            
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
                        logger.debug(f"Created unique constraint on {label}.{property_name}")
                        
                except Exception as e:
                    error_msg = str(e).lower()
                    if "already exists" in error_msg or "pending" in error_msg:
                        logger.debug(f"Constraint already exists for {label}.{property_name}")
                    else:
                        logger.warning(f"Constraint creation failed for {label}.{property_name}: {e}")
            
            # Create vector indexes if embedding provider is available
            if self.embedding_provider:
                self._create_vector_indexes()
            
            logger.info(f"Created {self.stats['indexes_created']} indexes and {self.stats['constraints_created']} constraints")
                        
        except Exception as e:
            logger.error(f"Error creating indexes/constraints: {e}")
    
    def _create_vector_indexes(self):
        """Create vector indexes for embeddings."""
        vector_fields = self.config.get('embedding', {}).get('vector_fields', [])
        logger.info(f"Creating {len(vector_fields)} vector indexes...")
        
        for field_config in vector_fields:
            try:
                entity_pattern = field_config['entity_pattern']
                attribute = field_config['attribute']
                
                options = {
                    'dimension': self.embedding_provider.dimensions,
                    'similarityFunction': field_config.get('similarityFunction', 'cosine'),
                    'M': field_config.get('M', 16),
                    'efConstruction': field_config.get('efConstruction', 200),
                    'efRuntime': field_config.get('efRuntime', 10)
                }
                
                # Build options string properly
                options_str = ", ".join([f"{k}: {v}" if isinstance(v, (int, float)) else f"{k}: '{v}'" for k, v in options.items()])
                
                query = f"CREATE VECTOR INDEX FOR {entity_pattern} ON ({attribute}) OPTIONS {{{options_str}}}"
                
                self.graph.query(query)
                self.stats['vector_indexes_created'] += 1
                logger.info(f"Created vector index: {entity_pattern} ON {attribute}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if "already exists" in error_msg:
                    logger.debug(f"Vector index already exists: {entity_pattern}")
                else:
                    logger.error(f"Error creating vector index: {e}")
    
    def _generate_embeddings_for_text_fields(self, df: pd.DataFrame, text_fields: List[str]) -> Dict[str, List[List[float]]]:
        """Generate embeddings for specified text fields with persistent file-based caching across runs."""
        if not self.embedding_provider:
            return {}
        
        # Create cache filename based on model and dimensions
        cache_filename = f"embedding_cache_{self.embedding_provider.model_name}_{self.embedding_provider.dimensions}d.pkl"
        
        # Load existing cache if it exists
        persistent_cache = {}
        if os.path.exists(cache_filename):
            try:
                with open(cache_filename, 'rb') as f:
                    persistent_cache = pickle.load(f)
                logger.info(f"Loaded {len(persistent_cache)} cached embeddings from {cache_filename}")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
                persistent_cache = {}
        
        embeddings = {}
        cache_updated = False
        
        for field in text_fields:
            if field not in df.columns:
                logger.warning(f"Field '{field}' not found in DataFrame columns: {list(df.columns)}")
                continue
            
            logger.info(f"Generating embeddings for field: {field}")
            
            # Get all texts and clean them
            texts = df[field].fillna('').astype(str).tolist()
            
            # Create hash-based keys for consistent caching
            uncached_texts = []
            text_to_hash = {}
            
            for text in texts:
                # Create hash of text for consistent cache key
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                text_to_hash[text] = text_hash
                
                if text_hash not in persistent_cache and text not in uncached_texts:
                    uncached_texts.append(text)
            
            cache_hits = len(texts) - len(uncached_texts)
            
            logger.info(f"Field '{field}': {len(texts)} total rows, {len(uncached_texts)} need embedding, {cache_hits} from persistent cache")
            
            try:
                # Generate embeddings only for uncached texts
                if uncached_texts:
                    new_embeddings = self.embedding_provider.embed_texts(uncached_texts)
                    
                    # Add to persistent cache
                    for i, text in enumerate(uncached_texts):
                        text_hash = text_to_hash[text]
                        persistent_cache[text_hash] = new_embeddings[i]
                    
                    cache_updated = True
                    self.stats['embeddings_created'] += len(uncached_texts)
                    logger.info(f"Generated {len(uncached_texts)} new embeddings, {cache_hits} reused from cache")
                else:
                    logger.info(f"All texts for field '{field}' found in persistent cache - no API calls needed!")
                
                # Update cache hit statistics
                self.stats['embeddings_from_cache'] += cache_hits
                
                # Build final embeddings list using cache
                field_embeddings = []
                for text in texts:
                    text_hash = text_to_hash[text]
                    field_embeddings.append(persistent_cache[text_hash])
                
                embeddings[field] = field_embeddings
                
            except Exception as e:
                logger.error(f"Error generating embeddings for field {field}: {e}")
                zero_vector = [0.0] * self.embedding_provider.dimensions
                embeddings[field] = [zero_vector] * len(texts)
        
        # Save updated cache if we added new embeddings
        if cache_updated:
            try:
                with open(cache_filename, 'wb') as f:
                    pickle.dump(persistent_cache, f)
                logger.info(f"Saved {len(persistent_cache)} embeddings to persistent cache: {cache_filename}")
            except Exception as e:
                logger.warning(f"Could not save embedding cache: {e}")
        
        return embeddings
    
    def _get_unique_key_field(self, node_label: str) -> Optional[str]:
        """Get the unique key field for a node label from constraints configuration."""
        constraints = self.config.get('constraints', [])
        
        logger.debug(f"Looking for unique constraint for label '{node_label}'")
        logger.debug(f"Available constraints: {constraints}")
        
        for constraint in constraints:
            constraint_label = constraint.get('label', '')
            constraint_property = constraint.get('property', '')
            constraint_type = constraint.get('type', 'UNIQUE')
            
            logger.debug(f"Checking constraint: label='{constraint_label}', property='{constraint_property}', type='{constraint_type}'")
            
            if constraint_label == node_label and constraint_type == 'UNIQUE':
                logger.info(f"Found unique constraint for {node_label}: {constraint_property}")
                return constraint_property
        
        logger.debug(f"No unique constraint found for label '{node_label}'")
        return None
    
    def _check_for_duplicates(self, df: pd.DataFrame, field_mappings: Dict, unique_key_field: str) -> Dict:
        """Check for duplicate key values and return statistics."""
        duplicate_stats = {
            'total_rows': len(df),
            'unique_keys': 0,
            'duplicate_rows': 0,
            'duplicate_keys': []
        }
        
        # Find the CSV field that maps to the unique key
        unique_csv_field = None
        for csv_field, graph_field in field_mappings.items():
            if graph_field == unique_key_field:
                unique_csv_field = csv_field
                break
        
        if not unique_csv_field or unique_csv_field not in df.columns:
            return duplicate_stats
        
        # Check for duplicates
        key_values = df[unique_csv_field].dropna()
        unique_values = key_values.drop_duplicates()
        
        duplicate_stats['unique_keys'] = len(unique_values)
        duplicate_stats['duplicate_rows'] = len(df) - len(unique_values)
        
        if duplicate_stats['duplicate_rows'] > 0:
            # Find the actual duplicate values
            duplicates = key_values[key_values.duplicated()].unique().tolist()
            duplicate_stats['duplicate_keys'] = duplicates[:10]  # Show first 10 duplicates
        
        return duplicate_stats

    def _process_nodes(self, file_config: Dict) -> int:
        """Process nodes with embeddings and duplicate handling."""
        csv_file = self.csv_path / file_config['file']
        
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return 0
        
        logger.info(f"Processing node file: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
            
            # Data quality analysis
            logger.info("Analyzing data quality...")
            quality_report = self.data_quality_analyzer.analyze_dataframe(df)
            self.stats['data_quality_reports'][file_config['file']] = quality_report
            logger.info(f"Data quality score: {quality_report['quality_score']:.2f}")
            
            node_label = file_config['node_label']
            field_mappings = file_config['field_mappings']
            
            # Check if this node type has a unique constraint
            unique_key_field = self._get_unique_key_field(node_label)
            use_merge = unique_key_field is not None
            
            # Additional fallback: check if any field mapping looks like an ID field
            if not use_merge:
                for csv_field, graph_field in field_mappings.items():
                    if graph_field.lower() in ['id', 'reportid', 'user_id', 'product_id', 'category_id', 'review_id']:
                        logger.warning(f"No unique constraint found for {node_label}, but detected ID field '{graph_field}'")
                        logger.warning(f"Enabling MERGE strategy as safety measure for potential ID field")
                        unique_key_field = graph_field
                        use_merge = True
                        break
            
            if use_merge:
                logger.info(f"Using MERGE strategy for {node_label} nodes (unique constraint on {unique_key_field})")
                
                # Check for duplicates in the data
                duplicate_stats = self._check_for_duplicates(df, field_mappings, unique_key_field)
                
                if duplicate_stats['duplicate_rows'] > 0:
                    logger.warning(f"Found {duplicate_stats['duplicate_rows']} duplicate rows for {node_label}")
                    logger.warning(f"Will keep only the last occurrence of each {unique_key_field}")
                    logger.warning(f"Example duplicate keys: {duplicate_stats['duplicate_keys']}")
                    
                    # Remove duplicates, keeping the last occurrence
                    unique_csv_field = None
                    for csv_field, graph_field in field_mappings.items():
                        if graph_field == unique_key_field:
                            unique_csv_field = csv_field
                            break
                    
                    if unique_csv_field and unique_csv_field in df.columns:
                        original_len = len(df)
                        # Remove rows with null unique keys first
                        df = df.dropna(subset=[unique_csv_field])
                        df = df.drop_duplicates(subset=[unique_csv_field], keep='last')
                        removed_count = original_len - len(df)
                        self.stats['duplicate_rows_removed'] += removed_count
                        self.stats['unique_constraint_violations_avoided'] += removed_count
                        logger.info(f"Removed {removed_count} duplicate rows, processing {len(df)} unique records")
                    else:
                        logger.error(f"Could not find CSV field for unique key {unique_key_field}")
                        logger.error(f"Available field mappings: {field_mappings}")
            else:
                logger.info(f"Using CREATE strategy for {node_label} nodes (no unique constraints detected)")
            
            # Generate embeddings if configured
            embeddings = {}
            embedding_fields = file_config.get('embedding_fields', [])
            if embedding_fields and self.embedding_provider:
                logger.info(f"Processing embedding fields: {embedding_fields}")
                embeddings = self._generate_embeddings_for_text_fields(df, embedding_fields)
            
            batch_size = file_config.get('batch_size', 1000)
            nodes_created = 0
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            logger.info(f"Creating {node_label} nodes in {total_batches} batches...")
            
            # Process in batches
            for i in range(0, len(df), batch_size):
                batch_num = i // batch_size + 1
                batch = df.iloc[i:i + batch_size]
                
                logger.debug(f"Processing batch {batch_num}/{total_batches}")
                
                statements = []
                for idx, (_, row) in enumerate(batch.iterrows()):
                    actual_idx = i + idx
                    
                    # Build basic properties
                    properties = self._build_cypher_properties(row, field_mappings)
                    
                    # Add embeddings as vector properties
                    for field, field_embeddings in embeddings.items():
                        if actual_idx < len(field_embeddings):
                            embedding = field_embeddings[actual_idx]
                            # Ensure embedding is JSON serializable
                            embedding = convert_to_json_serializable(embedding)
                            vector_field_name = f"{field}_embedding"
                            vector_str = f"vecf32({json.dumps(embedding)})"
                            # Remove closing brace and add vector property
                            if properties.endswith('}'):
                                if properties == '{}':
                                    properties = f"{{{vector_field_name}: {vector_str}}}"
                                else:
                                    properties = properties[:-1] + f", {vector_field_name}: {vector_str}" + "}"
                    
                    if use_merge and unique_key_field:
                        # Use MERGE for nodes with unique constraints
                        # Build the unique key property for matching
                        unique_value = None
                        unique_csv_field = None
                        
                        for csv_field, graph_field in field_mappings.items():
                            if graph_field == unique_key_field and csv_field in row.index:
                                unique_value = self._sanitize_value(row[csv_field])
                                unique_csv_field = csv_field
                                break
                        
                        if unique_value is not None and unique_value != "":
                            if isinstance(unique_value, str):
                                match_clause = f"{unique_key_field}: '{unique_value}'"
                            else:
                                match_clause = f"{unique_key_field}: {unique_value}"
                            
                            # Build SET clause with all properties
                            set_properties = properties
                            if set_properties == "{}":
                                set_properties = f"{{{match_clause}}}"
                            else:
                                # Merge the unique key into the properties if not already there
                                if unique_key_field not in set_properties:
                                    set_properties = set_properties[:-1] + f", {match_clause}" + "}"
                            
                            statements.append(f"MERGE (n{nodes_created}:{node_label} {{{match_clause}}}) SET n{nodes_created} = {set_properties}")
                        else:
                            logger.warning(f"Skipping row with null/empty unique key {unique_key_field} (CSV field: {unique_csv_field})")
                            continue
                    else:
                        # Use CREATE for nodes without unique constraints
                        statements.append(f"CREATE (n{nodes_created}:{node_label} {properties})")
                    
                    nodes_created += 1
                
                # Execute batch
                if statements:
                    query = " ".join(statements)
                    try:
                        self.graph.query(query)
                        logger.debug(f"Processed batch {batch_num}/{total_batches} ({len(statements)} nodes)")
                    except Exception as e:
                        logger.error(f"Error processing batch of {node_label} nodes: {e}")
                        logger.error(f"Batch info: use_merge={use_merge}, unique_key_field={unique_key_field}")
                        logger.error(f"Failed query snippet: {query[:500]}...")
                        
                        # If it's still a constraint violation, provide more helpful error message
                        if "unique constraint violation" in str(e).lower():
                            logger.error(f"Unique constraint violation detected. Debug info:")
                            logger.error(f"- Node label: {node_label}")
                            logger.error(f"- Detected unique key field: {unique_key_field}")
                            logger.error(f"- Using MERGE strategy: {use_merge}")
                            logger.error(f"- Available constraints in config: {self.config.get('constraints', [])}")
                            
                            # Try to extract which field is causing the violation
                            error_msg = str(e)
                            logger.error(f"Full error message: {error_msg}")
                            
                            if "reportid" in error_msg.lower():
                                logger.error("The violation is on 'reportid' field. Check your configuration:")
                                logger.error("Make sure your constraints section includes:")
                                logger.error('{"label": "Report", "property": "reportid", "type": "UNIQUE"}')
                        
                        self.stats['errors'].append(f"Node processing error in {csv_file}: {e}")
                        
                        # Try to process individual statements to identify the problematic one
                        if len(statements) > 1:
                            logger.info("Attempting to process statements individually to identify the problem...")
                            for i, stmt in enumerate(statements):
                                try:
                                    self.graph.query(stmt)
                                    logger.debug(f"Statement {i+1} executed successfully")
                                except Exception as stmt_error:
                                    logger.error(f"Problem statement {i+1}: {stmt}")
                                    logger.error(f"Error: {stmt_error}")
                                    break
            
            logger.info(f"Successfully processed {nodes_created} {node_label} nodes")
            return nodes_created
            
        except Exception as e:
            logger.error(f"Error processing nodes from {csv_file}: {e}")
            self.stats['errors'].append(f"Node processing error {csv_file}: {e}")
            return 0
    
    def _build_cypher_properties(self, row: pd.Series, field_mappings: Dict) -> str:
        """Build Cypher property string from pandas row."""
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
        """Sanitize and convert values for Cypher queries."""
        if pd.isna(value) or value is None:
            return None
        
        # Convert numpy types to native Python types
        value = convert_to_json_serializable(value)
        
        if isinstance(value, str):
            # Escape single quotes and handle special characters
            return value.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        
        if isinstance(value, (int, float)):
            return value
        
        if isinstance(value, bool):
            return str(value).lower()
        
        # Convert other types to string and escape
        str_value = str(value)
        return str_value.replace("'", "\\'").replace('"', '\\"')
    
    def _process_relationships(self, file_config: Dict) -> int:
        """Process relationships from CSV file."""
        csv_file = self.csv_path / file_config['file']
        
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return 0
        
        logger.info(f"Processing relationship file: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} relationships from {csv_file.name}")
            
            relationships_created = 0
            batch_size = file_config.get('batch_size', 1000)
            rel_config = file_config['relationship']
            rel_type = rel_config['type']
            
            logger.info(f"Creating {rel_type} relationships...")
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                
                for idx, row in batch.iterrows():
                    # Check if foreign key values are not null before creating relationship
                    source_key = row[rel_config['source']['csv_field']]
                    target_key = row[rel_config['target']['csv_field']]
                    
                    if pd.isna(source_key) or pd.isna(target_key) or source_key is None or target_key is None:
                        logger.debug(f"Skipping relationship due to null key: source={source_key}, target={target_key}")
                        continue
                    
                    # Build MATCH clauses for source and target nodes
                    source_match = self._build_match_clause(row, rel_config['source'], "source")
                    target_match = self._build_match_clause(row, rel_config['target'], "target")
                    
                    # Build relationship properties
                    rel_properties = ""
                    if 'properties' in rel_config:
                        props = self._build_cypher_properties(row, rel_config['properties'])
                        if props != "{}":
                            rel_properties = f" {props}"
                    
                    # Create relationship query
                    query = f"""
                    MATCH {source_match}
                    MATCH {target_match}
                    MERGE (source)-[r:{rel_type}{rel_properties}]->(target)
                    """
                    
                    try:
                        result = self.graph.query(query)
                        relationships_created += 1
                        
                        if relationships_created % 100 == 0:
                            logger.debug(f"Created {relationships_created} relationships so far...")
                            
                    except Exception as e:
                        logger.warning(f"Error creating relationship: {e}")
                        self.stats['errors'].append(f"Relationship creation error: {e}")
            
            logger.info(f"Successfully created {relationships_created} {rel_type} relationships")
            return relationships_created
            
        except Exception as e:
            logger.error(f"Error processing relationships from {csv_file}: {e}")
            self.stats['errors'].append(f"Relationship processing error {csv_file}: {e}")
            return 0
    
    def _build_match_clause(self, row: pd.Series, node_config: Dict, var_name: str = "n") -> str:
        """Build MATCH clause for finding nodes in relationships."""
        label = node_config['label']
        key_field = node_config['key_field']
        csv_field = node_config['csv_field']
        
        key_value = self._sanitize_value(row[csv_field])
        
        if isinstance(key_value, str):
            return f"({var_name}:{label} {{{key_field}: '{key_value}'}})"
        else:
            return f"({var_name}:{label} {{{key_field}: {key_value}}})"
    
    def _generate_profile_report(self) -> Dict:
        """Generate comprehensive profiling report."""
        logger.info("Generating profile report...")
        
        try:
            # Get graph statistics
            node_count_query = "MATCH (n) RETURN count(n) as node_count"
            relationship_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
            
            node_result = self.graph.query(node_count_query)
            rel_result = self.graph.query(relationship_count_query)
            
            total_nodes = node_result.result_set[0][0] if node_result.result_set else 0
            total_relationships = rel_result.result_set[0][0] if rel_result.result_set else 0
            
            # Convert to JSON serializable types
            total_nodes = convert_to_json_serializable(total_nodes)
            total_relationships = convert_to_json_serializable(total_relationships)
            
            # Get node type distribution
            node_types_query = "MATCH (n) RETURN labels(n) as label, count(n) as count ORDER BY count DESC"
            node_types_result = self.graph.query(node_types_query)
            
            node_distribution = {}
            for row in node_types_result.result_set:
                label = row[0][0] if row[0] else 'Unknown'
                count = convert_to_json_serializable(row[1])
                node_distribution[label] = count
            
            # Get relationship type distribution
            rel_types_query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC"
            rel_types_result = self.graph.query(rel_types_query)
            
            rel_distribution = {}
            for row in rel_types_result.result_set:
                rel_type = row[0]
                count = convert_to_json_serializable(row[1])
                rel_distribution[rel_type] = count
            
            # Calculate graph density
            graph_density = 0
            if total_nodes > 1:
                graph_density = total_relationships / (total_nodes * (total_nodes - 1))
            graph_density = convert_to_json_serializable(graph_density)
            
            # Enhanced embedding analytics with caching stats
            embedding_analytics = {}
            if self.embedding_provider:
                total_embeddings_generated = self.stats['embeddings_created']
                total_embeddings_from_cache = self.stats['embeddings_from_cache']
                total_embeddings_needed = total_embeddings_generated + total_embeddings_from_cache
                
                cache_hit_rate = 0
                if total_embeddings_needed > 0:
                    cache_hit_rate = total_embeddings_from_cache / total_embeddings_needed
                
                embedding_analytics = {
                    'total_embeddings_generated': convert_to_json_serializable(total_embeddings_generated),
                    'total_embeddings_from_cache': convert_to_json_serializable(total_embeddings_from_cache),
                    'cache_hit_rate': convert_to_json_serializable(cache_hit_rate),
                    'vector_dimensions': convert_to_json_serializable(self.embedding_provider.dimensions),
                    'model_name': self.embedding_provider.model_name,
                    'vector_indexes': convert_to_json_serializable(self.stats['vector_indexes_created']),
                    'estimated_api_cost_savings': f"${(total_embeddings_from_cache * 0.00013):.4f}" if total_embeddings_from_cache > 0 else "$0.0000"
                }
            
            profile_report = {
                'summary': {
                    'total_nodes': total_nodes,
                    'total_relationships': total_relationships,
                    'graph_density': graph_density,
                    'processing_time_seconds': convert_to_json_serializable(self.stats['processing_time']),
                    'files_processed': convert_to_json_serializable(self.stats['files_processed'])
                },
                'distributions': {
                    'nodes': node_distribution,
                    'relationships': rel_distribution
                },
                'embeddings': embedding_analytics,
                'data_quality': convert_to_json_serializable(self.stats['data_quality_reports']),
                'performance_stats': convert_to_json_serializable(self.stats),
                'timestamp': datetime.now().isoformat()
            }
            
            # Ensure entire report is JSON serializable
            return convert_to_json_serializable(profile_report)
            
        except Exception as e:
            logger.error(f"Error generating profile report: {e}")
            return convert_to_json_serializable({'error': str(e), 'timestamp': datetime.now().isoformat()})
    
    def convert(self):
        """Main conversion process."""
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING CSV TO FALKORDB CONVERSION WITH OPTIMIZED EMBEDDINGS")
            logger.info("=" * 60)
            
            # Create indexes, constraints, and vector indexes
            self._create_indexes_and_constraints()
            
            # Process node files
            node_files = self.config.get('node_files', [])
            logger.info(f"Processing {len(node_files)} node file(s)...")
            
            for file_config in node_files:
                nodes_created = self._process_nodes(file_config)
                self.stats['nodes_created'] += nodes_created
                self.stats['files_processed'] += 1
            
            # Process relationship files
            relationship_files = self.config.get('relationship_files', [])
            logger.info(f"Processing {len(relationship_files)} relationship file(s)...")
            
            for file_config in relationship_files:
                relationships_created = self._process_relationships(file_config)
                self.stats['relationships_created'] += relationships_created
                self.stats['files_processed'] += 1
            
            # Calculate processing time
            self.stats['processing_time'] = time.time() - start_time
            
            # Generate profile report
            profile_report = self._generate_profile_report()
            
            # Save profile report
            profile_path = f"{self.graph_name}_profile_report.json"
            with open(profile_path, 'w') as f:
                json.dump(profile_report, f, indent=2, cls=JSONEncoder)
            
            logger.info("=" * 60)
            logger.info("CONVERSION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Profile report saved to: {profile_path}")
            logger.info(f"Total nodes created: {self.stats['nodes_created']}")
            logger.info(f"Total relationships created: {self.stats['relationships_created']}")
            logger.info(f"Total embeddings generated: {self.stats['embeddings_created']}")
            logger.info(f"Total embeddings from cache: {self.stats['embeddings_from_cache']}")
            
            if self.stats['duplicate_rows_removed'] > 0:
                logger.info(f"Duplicate rows removed: {self.stats['duplicate_rows_removed']}")
                logger.info(f"Unique constraint violations avoided: {self.stats['unique_constraint_violations_avoided']}")
            
            if self.stats['embeddings_from_cache'] > 0:
                estimated_savings = self.stats['embeddings_from_cache'] * 0.00013
                logger.info(f"Estimated API cost savings: ${estimated_savings:.4f}")
            
            logger.info(f"Vector indexes created: {self.stats['vector_indexes_created']}")
            logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
            
            if self.stats['errors']:
                logger.warning(f"Encountered {len(self.stats['errors'])} errors during processing")
                for error in self.stats['errors']:
                    logger.warning(f"Error: {error}")
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if self.db:
                try:
                    self.db.close()
                    logger.info("Database connection closed")
                except:
                    pass
            if self.redis_client:
                try:
                    self.redis_client.close()
                    logger.info("Redis connection closed")
                except:
                    pass


def main():
    """Main entry point for the script."""
    print("FalkorDB CSV to Knowledge Graph Converter with Optimized Embeddings")
    print("=" * 70)
    
    parser = argparse.ArgumentParser(
        description="Convert CSV files to FalkorDB graph database with optimized OpenAI embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python graph_converter.py ecommerce_graph ./csv_files ./config.json
    python graph_converter.py social_network /path/to/csvs /path/to/config.json
    
Features:
    - Persistent embedding caching (saves costs across runs)
    - Duplicate text detection and optimization
    - Comprehensive data quality analysis
    - Vector similarity search support
        """
    )
    
    parser.add_argument('graph_name', help='Name of the graph to create in FalkorDB')
    parser.add_argument('csv_path', help='Path to directory containing CSV files')
    parser.add_argument('config_path', help='Path to configuration JSON file')
    
    try:
        args = parser.parse_args()
        
        # Validate arguments
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV directory not found: {args.csv_path}")
            sys.exit(1)
        
        if not os.path.exists(args.config_path):
            print(f"Error: Configuration file not found: {args.config_path}")
            sys.exit(1)
        
        # Show what we're about to do
        print(f"Graph name: {args.graph_name}")
        print(f"CSV path: {args.csv_path}")
        print(f"Config path: {args.config_path}")
        print()
        
        # Create and run converter
        converter = GraphConverter(args.graph_name, args.csv_path, args.config_path)
        converter.convert()
        
        print(f"\n✅ Successfully converted CSV data to FalkorDB graph: {args.graph_name}")
        print("📁 Check the generated profile report for detailed statistics")
        print("💾 Embedding cache saved for future runs (cost optimization)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()