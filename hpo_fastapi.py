# -*- coding: utf-8 -*-
"""
HPO Extraction FastAPI Application
REST API for Human Phenotype Ontology (HPO) term extraction from clinical notes
"""

import os
import sys
import time
import json
import unicodedata
import re
import traceback
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import tiktoken
import pandas as pd
import numpy as np
import requests
import faiss
from fastembed import TextEmbedding
from rapidfuzz import fuzz as rfuzz
from sentence_transformers import SentenceTransformer
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from llm_clients import OpenAICompatibleClient, GeminiNativeClient
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
FLAG_FILE = "GPT-5-mini.flag"
# ======================= Logger Class =======================
class Logger:
    def __init__(self):
        self.printed_messages = set()

    def log(self, msg, once=False):
        if once:
            msg_hash = hash(msg)
            if msg_hash in self.printed_messages:
                return
            self.printed_messages.add(msg_hash)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

app_logger = Logger()

# ======================= Utility Functions =======================
PAT = re.compile(r'\(.*?\)')

def clean_text(txt: str) -> str:
    return PAT.sub('', txt or '').replace('_', ' ').lower().strip()

def clean_note(text: str) -> str:
    text = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def initialize_embeddings_model(use_sbert: bool = True, 
                               sbert_model: str = 'pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb',
                               bge_model: str = 'BAAI/bge-small-en-v1.5'):
    try:
        if use_sbert:
            return SentenceTransformer(sbert_model)
        return TextEmbedding(model_name=bge_model)
    except Exception as e:
        logger.error(f"Could not initialize embedding model: {e}")
        raise

def load_vector_db(meta_path: str = 'hpo_meta.json', vec_path: str = 'hpo_embedded.npz'):
    if not os.path.exists(meta_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"DB files not found: {meta_path}, {vec_path}")

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            combined = json.load(f)
            constants = combined.get('constants', {})
            entries = combined.get('entries', [])
    except Exception as e:
        raise Exception(f"Could not load metadata JSON: {e}")

    try:
        arr = np.load(vec_path)
        emb_matrix = arr['emb'].astype(np.float32)
    except Exception as e:
        raise Exception(f"Could not load embedding npz: {e}")

    if len(entries) != emb_matrix.shape[0]:
        logger.warning(f"Metadata entries count and embedding rows mismatch ({len(entries)} vs {emb_matrix.shape[0]})")

    docs = []
    for entry, vec in zip(entries, emb_matrix):
        hp_id = entry.get('hp_id')
        const = constants.get(hp_id, {})
        doc = {
            'hp_id': hp_id,
            'info': entry.get('info'),
            'lineage': const.get('lineage'),
            'organ_system': const.get('organ_system'),
            'direction': entry.get('direction'),
            'depth': const.get('depth'),
            'parent_count': const.get('parent_count'),
            'child_count': const.get('child_count'),
            'descendant_count': const.get('descendant_count'),
            'embedding': vec
        }
        docs.append(doc)

    return docs, emb_matrix

def create_faiss_index(emb_matrix: np.ndarray, metric: str = 'cosine'):
    dim = emb_matrix.shape[1]
    if metric == 'cosine':
        faiss.normalize_L2(emb_matrix)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(emb_matrix)
    return index

def embed_query(text: str, model, metric: str = 'cosine'):
    if hasattr(model, 'encode'):
        vec = model.encode(text, convert_to_numpy=True)
    else:
        vec = np.array(list(model.embed([text]))[0], dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if metric == 'cosine':
        faiss.normalize_L2(vec)
    return vec

def _collect_metadata_best(phrase: str, query_vec: np.ndarray, index: faiss.Index, 
                          docs: List[Dict[str, Any]], top_k: int = 500,
                          similarity_threshold: float = 0.35, min_unique: int = 15, 
                          max_unique: int = 20) -> List[Dict[str, Any]]:
    clean_tokens = set(re.findall(r'\w+', phrase.lower()))
    dists, idxs = index.search(query_vec, top_k)
    sims, indices = dists[0], idxs[0]

    seen_hp = set()
    results = []

    for sim, idx in sorted(zip(sims, indices), key=lambda x: x[0], reverse=True):
        if len(results) >= max_unique:
            break
        doc = docs[idx]
        hp = doc.get('hp_id')
        if not hp or hp in seen_hp:
            continue

        info = doc.get('info', '') or ''
        token_overlap = bool(clean_tokens & set(re.findall(r'\w+', info.lower())))

        if token_overlap or sim >= similarity_threshold or len(results) < min_unique:
            seen_hp.add(hp)
            results.append({
                'hp_id': hp,
                'phrase': info,
                'definition': doc.get('definition'),
                'organ_system': doc.get('organ_system'),
                'similarity': float(sim)
            })

    return results

def clean_and_parse(s: str):
    try:
        m = re.search(r'\{.*\}', s, flags=re.S)
        js_str = m.group(0) if m else s.strip()
        return json.loads(js_str)
    except Exception:
        return None

def extract_findings(response: str) -> list:
    if not response:
        return []
    parsed = clean_and_parse(response)
    if not isinstance(parsed, dict):
        return []
    return parsed.get("phenotypes", [])

def process_findings(findings, clinical_note: str, embeddings_model, index, docs, 
                    metric: str = 'cosine', keep_top: int = 15):
    sentences = [s.strip() for s in clinical_note.split('.') if s.strip()]
    rows = []

    for f in findings:
        phrase = f.get('phrase', '').strip()
        category = f.get('category', '')
        if not phrase:
            continue

        qv = embed_query(phrase, embeddings_model, metric=metric)
        unique_metadata = _collect_metadata_best(
            phrase=phrase, query_vec=qv, index=index, docs=docs,
            top_k=500, similarity_threshold=0.35, min_unique=keep_top, max_unique=keep_top
        )

        fw = set(re.findall(r'\b\w+\b', phrase.lower()))
        best_sent, best_score = None, 0
        for s in sentences:
            sw = set(re.findall(r'\b\w+\b', s.lower()))
            score = len(fw & sw)
            if score > best_score:
                best_score, best_sent = score, s

        rows.append({
            'phrase': phrase,
            'category': category,
            'unique_metadata': unique_metadata,
            'original_sentence': best_sent,
            'patient_id': f.get('patient_id')
        })

    return pd.DataFrame(rows)

def process_clinical_note(clinical_note: str) -> List[Dict[str, Any]]:
    """Process a clinical note and return HPO terms"""
    global llm_client, embeddings_model, faiss_index, embedded_documents, system_message_I
    
    # Clean the clinical note
    clinical_note = clean_note(clinical_note)
    
    # Query the LLM
    raw = llm_client.query(clinical_note, system_message_I)
    
    # Extract findings
    findings = extract_findings(raw)
    if not findings:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                findings = parsed
        except Exception:
            pass

    if not findings:
        try:
            matches = re.findall(r'\{[^\}]*\}', raw)
            findings = []
            for m in matches:
                try:
                    d = json.loads(m)
                    if 'phrase' in d and 'category' in d:
                        findings.append(d)
                except Exception:
                    continue
        except Exception:
            pass

    # Filter for abnormal findings only
    findings = [f for f in findings if isinstance(f, dict) and f.get('category') == 'Abnormal']

    if not findings:
        return []

    # Process findings
    df = process_findings(findings, clinical_note, embeddings_model, faiss_index, embedded_documents)
    
    # Extract HPO terms using local mapping
    results = []
    for _, row in df.iterrows():
        phrase = row['phrase']
        category = row['category']
        metadata_list = row['unique_metadata'] or []
        
        # Simple HPO term extraction (using the best match from metadata)
        hpo_term = None
        if metadata_list:
            hpo_term = metadata_list[0].get('hp_id')  # Take the best match
        
        results.append({
            'phrase': phrase,
            'category': category,
            'hpo_id': hpo_term or 'No Candidate Fit'
        })
    
    return results

# ======================= Original Methods from Pipeline =======================
def load_prompts(file_path="system_prompts.json"):
    if not os.path.exists(file_path):
        logger.error(f"Prompt file '{file_path}' not found.")
        raise FileNotFoundError(f"Prompt file '{file_path}' not found.")
    with open(file_path, "r") as f:
        return json.load(f)

def check_and_initialize_llm():
    """
    Initialize LLM client based on the 'model_type' in the flag file.
    Dynamically chooses between 'openai' or 'gemini' clients.
    """
    
    if not os.path.exists(FLAG_FILE):
        raise FileNotFoundError(f"Flag file '{FLAG_FILE}' not found. Please run the initialization first.")

    with open(FLAG_FILE, "r") as f:
        config = json.load(f)

    # 從設定檔讀取 model_type
    model_type = config.get("model_type")

    # 讀取通用參數
    common_args = {
        "api_key": config["api_key"],
        "base_url": config.get("base_url"),
        "model_name": config.get("model_name"),
        "temperature": config.get("temperature", 0.7),
        "max_tokens_per_day": config.get("max_tokens_per_day", -1),
        "max_queries_per_minute": config.get("max_queries_per_minute", 3500),
        "max_tokens_per_minute": config.get("max_tokens_per_minute", 4000000)
    }

    client_class = None
    # 根據 model_type 來決定要用哪個 Client
    if model_type == "openai":
        logger.info("Initializing OpenAI-compatible LLM Client based on 'model_type'...")
        client_class = OpenAICompatibleClient
    elif model_type == "gemini":
        logger.info("Initializing Gemini Native LLM Client based on 'model_type'...")
        client_class = GeminiNativeClient
    else:
        # 如果 model_type 不存在或設定錯誤，則拋出例外
        raise ValueError(
            f"Invalid or missing 'model_type' in config file. "
            f"Must be 'openai' or 'gemini'. Got: {model_type}"
        )

    # 使用通用參數來實例化選擇的 Class
    return client_class(**common_args)

# ======================= Startup/Shutdown =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm_client, embeddings_model, faiss_index, embedded_documents, system_message_I, system_message_II
    
    logger.info("Starting HPO Extraction API...")
    
    try:
        # Load system prompts using original method
        prompts = load_prompts()
        system_message_I = prompts.get("system_message_I", "")
        system_message_II = prompts.get("system_message_II", "")
        
        # Initialize LLM client using flag file method
        llm_client = check_and_initialize_llm()
        
        # Initialize embeddings model
        embeddings_model = initialize_embeddings_model(use_sbert=True)
        
        # Load vector database
        docs, emb_matrix = load_vector_db(meta_path='hpo_meta.json', vec_path='hpo_embedded.npz')
        embedded_documents = docs
        
        # Create FAISS index
        faiss_index = create_faiss_index(emb_matrix, metric='cosine')
        
        logger.info("HPO Extraction API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize HPO Extraction API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down HPO Extraction API...")

# ======================= Pydantic Models =======================
class AnnotateRequest(BaseModel):
    text: str = Field(..., description="Clinical note text to extract HPO terms from", min_length=1)

class AnnotateResponse(BaseModel):
    text: str
    hpo_terms: str = Field(..., description="HPO terms in format: 'term1 (HP:xxxxx);term2 (HP:xxxxx);...'")
    hpo_ids: str = Field(..., description="HPO IDs only: 'HP:xxxxx;HP:xxxxx;...'")
    processing_time: float

class BatchAnnotateRequest(BaseModel):
    texts: List[str] = Field(..., description="List of clinical notes to process", min_items=1)

class BatchAnnotateResponse(BaseModel):
    results: List[AnnotateResponse]
    total_processing_time: float
    count: int

class HealthResponse(BaseModel):
    status: str
    model_info: Dict
    timestamp: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# ======================= FastAPI App =======================
app = FastAPI(
    title="HPO Extraction API",
    description="Human Phenotype Ontology (HPO) term extraction from clinical notes",
    version="1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= API Routes =======================
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information"""
    return {
        "service": "HPO Extraction API",
        "version": "1.0",
        "description": "HPO term extraction from clinical notes",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if llm_client is None or embeddings_model is None:
        raise HTTPException(status_code=503, detail="API not fully initialized")
    
    try:
        model_info = {
            "llm_model": llm_client.model_name,
            "embedding_model": "SentenceTransformer",
            "vector_db_loaded": embedded_documents is not None,
            "faiss_index_ready": faiss_index is not None
        }
        return HealthResponse(
            status="healthy",
            model_info=model_info,
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/annotate", response_model=AnnotateResponse)
async def annotate_text_endpoint(request: AnnotateRequest):
    """
    Extract HPO terms from a single clinical note
    """
    if llm_client is None or embeddings_model is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    try:
        start_time = time.time()
        
        # Process clinical note
        hpo_terms = process_clinical_note(request.text)
        
        processing_time = time.time() - start_time
        
        # Format output strings
        hpo_terms_str = ""
        hpo_ids_str = ""
        
        if hpo_terms:
            # Create formatted strings and remove duplicates
            term_parts = []
            hpo_id_parts = []
            seen_hpo_ids = set()
            
            for term in hpo_terms:
                phrase = term['phrase']
                hpo_id = term['hpo_id']
                
                # Only include valid HPO IDs (not "No Candidate Fit") and avoid duplicates
                if hpo_id and hpo_id != "No Candidate Fit" and hpo_id.startswith("HP:") and hpo_id not in seen_hpo_ids:
                    term_parts.append(f"{phrase} ({hpo_id})")
                    hpo_id_parts.append(hpo_id)
                    seen_hpo_ids.add(hpo_id)
            
            hpo_terms_str = ";".join(term_parts)
            hpo_ids_str = ";".join(hpo_id_parts)
        
        return AnnotateResponse(
            text=request.text,
            hpo_terms=hpo_terms_str,
            hpo_ids=hpo_ids_str,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")

@app.post("/annotate/batch", response_model=BatchAnnotateResponse)
async def annotate_batch_endpoint(request: BatchAnnotateRequest):
    """
    Extract HPO terms from multiple clinical notes
    """
    if llm_client is None or embeddings_model is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    if len(request.texts) > 100:
        logger.warning(f"Processing large batch of {len(request.texts)} texts")
    
    try:
        start_time = time.time()
        
        results = []
        for i, text in enumerate(request.texts):
            text_start_time = time.time()
            
            # Add delay between batch requests to avoid TPM limits
            if i > 0:  # Skip delay for first request
                delay = 60 / llm_client.max_queries_per_minute
                logger.info(f"Waiting {delay:.2f}s before processing text {i+1}/{len(request.texts)}")
                time.sleep(delay)
            
            hpo_terms = process_clinical_note(text)
            text_processing_time = time.time() - text_start_time
            
            # Format output strings
            hpo_terms_str = ""
            hpo_ids_str = ""
            
            if hpo_terms:
                # Create formatted strings and remove duplicates
                term_parts = []
                hpo_id_parts = []
                seen_hpo_ids = set()
                
                for term in hpo_terms:
                    phrase = term['phrase']
                    hpo_id = term['hpo_id']
                    
                    # Only include valid HPO IDs (not "No Candidate Fit") and avoid duplicates
                    if hpo_id and hpo_id != "No Candidate Fit" and hpo_id.startswith("HP:") and hpo_id not in seen_hpo_ids:
                        term_parts.append(f"{phrase} ({hpo_id})")
                        hpo_id_parts.append(hpo_id)
                        seen_hpo_ids.add(hpo_id)
                
                hpo_terms_str = ";".join(term_parts)
                hpo_ids_str = ";".join(hpo_id_parts)
            
            results.append(AnnotateResponse(
                text=text,
                hpo_terms=hpo_terms_str,
                hpo_ids=hpo_ids_str,
                processing_time=text_processing_time
            ))
            
            logger.info(f"Completed processing text {i+1}/{len(request.texts)}")
        
        total_processing_time = time.time() - start_time
        
        return BatchAnnotateResponse(
            results=results,
            total_processing_time=total_processing_time,
            count=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Batch annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch annotation failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current API configuration"""
    if llm_client is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    try:
        return {
            "model_info": {
                "llm_model": llm_client.model_name,
                "embedding_model": "SentenceTransformer",
                "max_tokens_per_day": llm_client.max_tokens_per_day,
                "tokens_used": llm_client.total_tokens_used
            },
            "vector_db_info": {
                "documents_loaded": len(embedded_documents) if embedded_documents else 0,
                "index_ready": faiss_index is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hpo_fastapi:app",
        host="192.168.5.77",
        port=8333,
        reload=False,
        log_level="info"
    )