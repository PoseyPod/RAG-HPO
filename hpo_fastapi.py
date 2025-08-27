# -*- coding: utf-8 -*-
"""
Simplified HPO Extraction FastAPI with Basic Caching
Minimal maintenance overhead while preserving essential functionality
"""

import os
import time
import json
import hashlib
import pickle
import unicodedata
import re
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llm_clients import OpenAICompatibleClient, GeminiNativeClient, OllamaClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================= Global Variables =======================
llm_client = None
embeddings_model = None
faiss_index = None
embedded_documents = None
system_message_I = ""
system_message_II = ""
system_message_double_check = ""
FLAG_FILE = "ollama.flag"

# ======================= Simple Cache =======================
class SimpleCache:
    """Lightweight caching with file persistence"""
    
    def __init__(self, cache_file: str = "llm_cache.pkl"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        try:
            with open(f"{self.cache_file}.tmp", 'wb') as f:
                pickle.dump(self.cache, f)
            os.replace(f"{self.cache_file}.tmp", self.cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, text: str, pipeline_key: str) -> Optional[Any]:
        key = hashlib.md5(f"{text}||{pipeline_key}".encode()).hexdigest()
        return self.cache.get(key)
    
    def set(self, text: str, pipeline_key: str, response: Any):
        key = hashlib.md5(f"{text}||{pipeline_key}".encode()).hexdigest()
        self.cache[key] = response
        # Save every 10 new entries to balance performance and persistence
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    def size(self) -> int:
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

# Initialize cache
cache = SimpleCache()

# ======================= Utility Functions =======================
def clean_note(text: str) -> str:
    """Basic text cleaning"""
    text = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _safe_json_loads(text: str) -> Optional[Dict]:
    """Safely parse JSON from a string, searching for the first valid object."""
    if not text:
        return None
    try:
        # First, try to parse the whole string
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, search for a JSON object within the string
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

def load_vector_db(meta_path: str = 'hpo_meta.json', vec_path: str = 'hpo_embedded.npz'):
    """Load HPO vector database"""
    if not os.path.exists(meta_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"DB files not found: {meta_path}, {vec_path}")

    with open(meta_path, 'r', encoding='utf-8') as f:
        combined = json.load(f)
        entries = combined.get('entries', [])

    arr = np.load(vec_path)
    emb_matrix = arr['emb'].astype(np.float32)

    docs = []
    for entry, vec in zip(entries, emb_matrix):
        docs.append({
            'hp_id': entry.get('hp_id'),
            'info': entry.get('info'),
            'embedding': vec
        })

    return docs, emb_matrix

def create_faiss_index(emb_matrix: np.ndarray):
    """Create FAISS index for similarity search"""
    dim = emb_matrix.shape[1]
    faiss.normalize_L2(emb_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    return index

def embed_query(text: str, model):
    """Embed query text"""
    vec = model.encode(text, convert_to_numpy=True)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def extract_hpo_terms(clinical_note: str) -> Dict[str, Any]:
    """
    Main processing function using a simplified two-step pipeline:
    1. Extract candidate phenotypes (system_message_I).
    2. Filter for 'Abnormal' category and map to HPO IDs (system_message_II).
    The LLM-based double-check step has been removed.
    """
    global llm_client, embeddings_model, faiss_index, embedded_documents
    # REMOVED system_message_double_check
    global system_message_I, system_message_II
    
    # Updated pipeline key for cache to reflect the new logic
    PIPELINE_KEY = "hpo_pipeline_v1.4_no_doublecheck" 
    
    # Clean input
    clinical_note = clean_note(clinical_note)
    
    # Check cache for the final result of the entire pipeline
    cached_response = cache.get(clinical_note, PIPELINE_KEY)
    if cached_response:
        logger.info("Using cached final result for the simplified pipeline.")
        return cached_response

    # Initialize result structure and thinking process tracking
    final_result = {'hpo_terms': [], 'thinking_process': []}
    
    # --- Step 1: Initial Phenotype Extraction ---
    logger.info("Step 1: Initial Phenotype Extraction")
    raw_response_I = llm_client.query(clinical_note, system_message_I)
    thinking_I, content_I = extract_thinking_from_content(raw_response_I)
    if thinking_I:
        final_result['thinking_process'].append(f"--- Step 1: Extraction ---\n{thinking_I}")

    initial_phenotypes = []
    parsed_I = _safe_json_loads(content_I)
    if parsed_I and 'phenotypes' in parsed_I:
        initial_phenotypes = parsed_I['phenotypes']
    
    if not initial_phenotypes:
        logger.warning("No initial phenotypes extracted in Step 1.")
        final_result['thinking_process'] = "\n\n".join(final_result['thinking_process'])
        return final_result

    # --- Step 2: Filter for 'Abnormal' Phenotypes (Double-Check step is removed) ---
    logger.info("Step 2: Skipping LLM Double-Check. Filtering for 'Abnormal' category phenotypes.")
    # phenotypes_to_map = [
    #     p for p in initial_phenotypes 
    #     if isinstance(p, dict) and p.get('category') == 'Abnormal'
    # ]
    #logger.info(f"Found {len(phenotypes_to_map)} 'Abnormal' phenotypes to map.")
    
    # pass all the phenotypes from the step 1 result
    phenotypes_to_map = [p for p in initial_phenotypes if isinstance(p, dict)]
    logger.info(f"Found {len(phenotypes_to_map)} phenotypes to map.")
    if not phenotypes_to_map:
        final_result['thinking_process'] = "\n\n".join(final_result['thinking_process'])
        return final_result
        
    # --- Step 3: Map Filtered Phenotypes to HPO IDs ---
    logger.info(f"Step 3: Mapping {len(phenotypes_to_map)} phenotypes to HPO.")
    final_hpo_terms = []
    for finding in phenotypes_to_map:
        phrase = finding.get('phrase', '').strip()
        if not phrase:
            continue
        
        query_vec = embed_query(phrase, embeddings_model)
        distances, indices = faiss_index.search(query_vec, 5) # Get top 5 candidates
        
        candidates = []
        if indices.size > 0:
            for idx in indices[0]:
                if idx >= 0:
                    match = embedded_documents[idx]
                    candidates.append({"term": match.get('info'), "id": match.get('hp_id')})
        
        if not candidates:
            logger.warning(f"No FAISS candidates found for phrase: '{phrase}'")
            continue

        mapper_input = {
            "phrase": phrase,
            "category": finding.get('category'),
            "candidates": candidates
        }
        
        mapper_response = llm_client.query(json.dumps(mapper_input, indent=2), system_message_II)
        thinking_II, content_II = extract_thinking_from_content(mapper_response)
        if thinking_II:
            final_result['thinking_process'].append(f"--- Step 3 Mapping for '{phrase}' ---\n{thinking_II}")

        parsed_map = _safe_json_loads(content_II)
        
        if parsed_map and 'hpo_id' in parsed_map and parsed_map['hpo_id']:
            final_hpo_terms.append({
                'phrase': phrase,
                'category': finding.get('category', 'Abnormal'),
                'hpo_id': parsed_map['hpo_id']
            })

    final_result['hpo_terms'] = final_hpo_terms
    final_result['thinking_process'] = "\n\n".join(final_result['thinking_process'])

    # Cache the final result of the entire pipeline
    cache.set(clinical_note, PIPELINE_KEY, final_result)
    logger.info("Cached new final result.")
    
    return final_result


def extract_thinking_from_content(content: str) -> tuple:
    """
    Extract thinking process from <think>...</think> tags in content.
    Returns (thinking_process, cleaned_content)
    """
    if not content:
        return "", ""
    
    # Handle if content is a JSON string representing the full API response
    try:
        data = json.loads(content)
        if "choices" in data and data["choices"]:
            message_content = data["choices"][0].get("message", {}).get("content", "")
            if message_content:
                content = message_content
    except (json.JSONDecodeError, TypeError):
        pass # It's not a JSON string, process as plain text

    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, content, re.DOTALL)
    
    if matches:
        thinking_process = matches[0].strip()
        cleaned_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
        return thinking_process, cleaned_content
    else:
        return "", content

def check_and_initialize_llm():
    """Initialize LLM client from config file"""
    if not os.path.exists(FLAG_FILE):
        raise FileNotFoundError(f"Flag file '{FLAG_FILE}' not found.")

    with open(FLAG_FILE, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type")
    common_args = {
        "api_key": config["api_key"],
        "base_url": config.get("base_url"),
        "model_name": config.get("model_name"),
        "temperature": config.get("temperature", 0.7),
        "max_tokens_per_day": config.get("max_tokens_per_day", -1),
        "max_queries_per_minute": config.get("max_queries_per_minute", 60),
        "max_tokens_per_minute": config.get("max_tokens_per_minute", 4000000),
        "think": config.get("think", False)  # Add think parameter for Ollama
    }

    if model_type == "openai":
        return OpenAICompatibleClient(**common_args)
    elif model_type == "gemini":
        return GeminiNativeClient(**common_args)
    elif model_type == "ollama":
        return OllamaClient(**common_args)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

# ======================= Startup/Shutdown =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, embeddings_model, faiss_index, embedded_documents
    # REMOVED system_message_double_check from globals
    global system_message_I, system_message_II
    
    logger.info("Starting HPO Extraction API...")
    
    try:
        # Load required prompts
        with open("system_prompts.json", "r") as f:
            prompts = json.load(f)
        system_message_I = prompts.get("system_message_I", "")
        system_message_II = prompts.get("system_message_II", "")
        
        # REMOVED loading of system_message_double_check
        if not all([system_message_I, system_message_II]):
            raise ValueError("system_message_I or system_message_II are missing from system_prompts.json")

        # Initialize components
        llm_client = check_and_initialize_llm()
        embeddings_model = SentenceTransformer('pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb')
        
        docs, emb_matrix = load_vector_db()
        embedded_documents = docs
        faiss_index = create_faiss_index(emb_matrix)
        
        logger.info(f"API initialized with cache size: {cache.size()}")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Save cache on shutdown
    cache._save_cache()
    logger.info("Shutting down...")

# ======================= Pydantic Models =======================
class AnnotateRequest(BaseModel):
    text: str = Field(..., min_length=1)

class AnnotateResponse(BaseModel):
    text: str
    hpo_terms: str
    hpo_ids: str
    processing_time: float
    cached: bool = False
    thinking_process: str = Field(default="", description="AI reasoning process from the multi-step pipeline")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)

class BatchResponse(BaseModel):
    results: List[AnnotateResponse]
    total_time: float
    cached_count: int
    count: int
    total_processing_time: float

# ======================= FastAPI App =======================
# Create app with lifespan
app = FastAPI(
    title="Simple HPO Extraction API",
    description="Lightweight HPO term extraction with caching and a multi-step validation pipeline.",
    version="1.2",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Simple HPO Extraction API",
        "version": "1.2",
        "cache_size": cache.size(),
        "pipeline_active": "multi-step"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cache_size": cache.size(),
        "timestamp": time.time()
    }

@app.post("/annotate", response_model=AnnotateResponse)
async def annotate(request: AnnotateRequest):
    """Extract HPO terms from clinical note"""
    try:
        start_time = time.time()
        
        # Check if cached
        was_cached = bool(cache.get(clean_note(request.text), "hpo_pipeline_v1.2"))
        
        # Process
        hpo_result = extract_hpo_terms(request.text)
        hpo_terms = hpo_result.get('hpo_terms', [])
        thinking_process = hpo_result.get('thinking_process', '')
        
        # Format results
        term_parts = []
        id_parts = []
        seen_ids = set()
        
        for term in hpo_terms:
            phrase = term['phrase']
            hpo_id = term['hpo_id']
            
            if hpo_id and hpo_id.startswith("HP:") and hpo_id not in seen_ids:
                term_parts.append(f"{phrase} ({hpo_id})")
                id_parts.append(hpo_id)
                seen_ids.add(hpo_id)
        
        return AnnotateResponse(
            text=request.text,
            hpo_terms=";".join(term_parts),
            hpo_ids=";".join(id_parts),
            processing_time=time.time() - start_time,
            cached=was_cached,
            thinking_process=thinking_process
        )
        
    except Exception as e:
        logger.error(f"Annotation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ... (The rest of the file, including batch processing and error handling, remains the same)
# Global error tracking
recent_errors = []

def log_error_text(text: str, error: str, index: int = None):
    """Log error text to memory for API access"""
    error_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'index': index,
        'error': str(error),
        'text_preview': text[:200] if text else "None",
        'text_length': len(text) if text else 0,
        'text_hash': hashlib.md5(text.encode() if text else b'').hexdigest()[:8]
    }
    recent_errors.append(error_info)
    # Keep only last 20 errors
    if len(recent_errors) > 20:
        recent_errors.pop(0)

def save_problematic_text(text: str, error: str, index: int = None):
    """Save problematic text to file for debugging"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"error_texts_{timestamp}.txt"
    
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"=== Error at index {index} at {timestamp} ===\n")
            f.write(f"Error: {error}\n")
            f.write(f"Text length: {len(text) if text else 0}\n")
            f.write(f"Text hash: {hashlib.md5(text.encode() if text else b'').hexdigest()[:8]}\n")
            f.write(f"Text content:\n{text if text else 'None'}\n")
            f.write("=" * 50 + "\n\n")
        logger.info(f"Saved problematic text to {filename}")
    except Exception as save_error:
        logger.error(f"Failed to save problematic text: {save_error}")

@app.post("/annotate/batch", response_model=BatchResponse)
async def annotate_batch(request: BatchRequest):
    """Extract HPO terms from multiple clinical notes - supports large batches"""
    try:
        start_time = time.time()
        results = []
        cached_count = 0
        
        total_texts = len(request.texts)
        logger.info(f"Starting batch processing of {total_texts} texts")
        
        PIPELINE_KEY = "hpo_pipeline_v1.2"

        for i, text in enumerate(request.texts):
            try:
                text_length = len(text) if text else 0
                logger.info(f"Processing text {i+1}/{total_texts} (length: {text_length} chars)")
                
                if text and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Text preview: '{text[:100]}...'")
                
                if not text or not text.strip():
                    logger.warning(f"Empty text at index {i+1}, skipping")
                    results.append(AnnotateResponse(
                        text=text or "", hpo_terms="", hpo_ids="",
                        processing_time=0, cached=False, thinking_process=""
                    ))
                    continue
                
                cleaned_text = clean_note(text)
                was_cached = bool(cache.get(cleaned_text, PIPELINE_KEY))

                if not was_cached and llm_client and hasattr(llm_client, 'max_queries_per_minute'):
                     # Simple delay to respect rate limits for non-cached items.
                    delay_time = 60.0 / llm_client.max_queries_per_minute
                    time.sleep(delay_time)

                text_start = time.time()
                if was_cached:
                    cached_count += 1
                
                hpo_result = extract_hpo_terms(text)
                hpo_terms = hpo_result.get('hpo_terms', [])
                thinking_process = hpo_result.get('thinking_process', "")
                
                term_parts, id_parts, seen_ids = [], [], set()
                for term in hpo_terms:
                    phrase = term.get('phrase', '')
                    hpo_id = term.get('hpo_id', '')
                    if hpo_id and hpo_id.startswith("HP:") and hpo_id not in seen_ids:
                        term_parts.append(f"{phrase} ({hpo_id})")
                        id_parts.append(hpo_id)
                        seen_ids.add(hpo_id)
                
                results.append(AnnotateResponse(
                    text=text,
                    hpo_terms=";".join(term_parts),
                    hpo_ids=";".join(id_parts),
                    processing_time=time.time() - text_start,
                    cached=was_cached,
                    thinking_process=thinking_process
                ))
                
                if (i + 1) % 10 == 0 or (i + 1) == total_texts:
                    logger.info(f"Processed {i + 1}/{total_texts} texts ({cached_count} from cache)")
                
                if (i + 1) % 50 == 0:
                    cache._save_cache()
                    logger.info(f"Cache saved at item {i + 1}")
                    
            except Exception as text_error:
                error_msg = str(text_error)
                logger.error(f"Error processing text {i+1}/{total_texts}: {error_msg}", exc_info=True)
                log_error_text(text, error_msg, i+1)
                save_problematic_text(text, error_msg, i+1)
                
                results.append(AnnotateResponse(
                    text=text or "", hpo_terms="ERROR: Processing failed", hpo_ids="ERROR",
                    processing_time=0, cached=False, thinking_process=f"Error: {error_msg}"
                ))
                continue
        
        total_processing_time = time.time() - start_time
        logger.info(f"Batch completed: {len(results)} results in {total_processing_time:.2f}s")
        
        return BatchResponse(
            results=results, total_time=total_processing_time,
            cached_count=cached_count, count=len(results),
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch processing completely failed: {e}", exc_info=True)
        cache._save_cache()
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/debug/recent-errors")
async def get_recent_errors():
    """Get recently failed texts for debugging"""
    return {
        "total_recent_errors": len(recent_errors),
        "errors": recent_errors
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the response cache"""
    cache.clear()
    return {"message": "Cache cleared"}

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cache_size": cache.size(),
        "cache_file_exists": os.path.exists(cache.cache_file)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8333,
        reload=False
    )