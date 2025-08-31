# -*- coding: utf-8 -*-
"""
Modified HPO Extraction FastAPI with New Pipeline
Three-step process: Extract -> Normalize -> Vector Retrieval
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
from typing import List
from langchain_llm_clients import LangchainOpenAIClient, LangchainGeminiClient, LangchainOllamaClient, LangchainVLLMClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================= Global Variables =======================
llm_client = None
embeddings_model = None
faiss_index = None
embedded_documents = None
model_type = None
system_message_extract = ""
system_message_normalize = ""


# --- [修改 1/4]: 新增全域變數來儲存設定 ---
app_config: Dict[str, Any] = {}
embedding_model_name = 'pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb'
#embedding_model_name = 'FremyCompany/BioLORD-2023'
#meta_path = 'deeprare_hpo_meta.json'
meta_path = 'hpo_meta.json'
#vec_path= 'deeprare_hpo_embedded.npz'
vec_path= 'hpo_embedded.npz'

system_prompt_path = "openai_system_prompts.json"
FLAG_FILE = "openai.flag"

# system_prompt_path = "gemini_system_prompts.json"
# FLAG_FILE = "gemini.flag"

# ====================== LLM Output Schema ======================#
class HPOPhenotype(BaseModel):
    HPO: str = Field(description="HPO ID in format HP:0000000")
    Phenotype: str = Field(description="Clinical phenotype description")

class HPOExtractionResult(BaseModel):
    """A container for a list of extracted HPO phenotypes."""
    phenotypes: List[HPOPhenotype] = Field(description="A list of HPO phenotypes extracted from the clinical text.")

class PhenotypeNormalization(BaseModel):
    original_term: str = Field(description="Original phenotype description")
    hpo_term: str = Field(description="Standardized HPO term in English, or 'none' if not found")

# ======================= Simple Cache =======================
# ... (此區塊程式碼不變)
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
            except: return {}
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
        if len(self.cache) % 10 == 0:
            self._save_cache()

    def size(self) -> int:
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

cache = SimpleCache()

# ======================= Utility Functions =======================
# ... (此區塊程式碼不變)
def clean_note(text: str) -> str:
    text = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _safe_json_loads(text: str) -> Optional[Dict]:
    if not text: return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    markdown_json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if markdown_json_match:
        json_content = markdown_json_match.group(1).strip()
        try: return json.loads(json_content)
        except json.JSONDecodeError: pass
    markdown_match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if markdown_match:
        json_content = markdown_match.group(1).strip()
        try: return json.loads(json_content)
        except json.JSONDecodeError: pass
    try: return json.loads(text)
    except json.JSONDecodeError:
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            try: return json.loads(array_match.group(0))
            except json.JSONDecodeError: pass
        object_match = re.search(r'\{.*\}', text, re.DOTALL)
        if object_match:
            try: return json.loads(object_match.group(0))
            except json.JSONDecodeError: pass
    return None

def load_vector_db(meta_path: str = 'hpo_meta.json', vec_path: str = 'hpo_embedded.npz'):
    if not os.path.exists(meta_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"DB files not found: {meta_path}, {vec_path}")
    with open(meta_path, 'r', encoding='utf-8') as f:
        combined = json.load(f)
        entries = combined.get('entries', [])
    arr = np.load(vec_path)
    emb_matrix = arr['emb'].astype(np.float32)
    docs = []
    for entry, vec in zip(entries, emb_matrix):
        docs.append({'hp_id': entry.get('hp_id'), 'info': entry.get('info'), 'embedding': vec})
    return docs, emb_matrix

def create_faiss_index(emb_matrix: np.ndarray):
    dim = emb_matrix.shape[1]
    faiss.normalize_L2(emb_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    return index

PAT_CLEAN = re.compile(r'\s*\([^)]*\)\s*')
def clean_query_text(txt: str) -> str:
    txt = PAT_CLEAN.sub(' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip().lower()
    txt = re.sub(r'[^\w\s]+$', '', txt)
    return txt

def embed_query(text: str, model):
    """Embed query text after cleaning."""
    # 在嵌入前，先進行清理
    cleaned_text = clean_query_text(text)
    
    vec = model.encode(cleaned_text, convert_to_numpy=True)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

# ======================= Updated HPO Extraction Function =======================
# ... (此區塊程式碼不變)
def extract_hpo_terms_structured(clinical_note: str) -> Dict[str, Any]:
    global llm_client, embeddings_model, faiss_index, embedded_documents
    global system_message_extract, system_message_normalize, model_type
    PIPELINE_KEY = "hpo_pipeline_v2.1_structured_output"
    clinical_note = clean_note(clinical_note)
    cached_response = cache.get(clinical_note, PIPELINE_KEY)
    if cached_response:
        logger.info("Using cached final result for the structured pipeline.")
        return cached_response
    final_result = {'hpo_terms': [], 'thinking_process': []}
    logger.info("Step 1: Extracting phenotypes using structured output")
    extracted_phenotypes = []
    try:
        if model_type == 'gemini':
            raw_response_extract = llm_client.query(clinical_note, system_message_extract, response_schema=HPOExtractionResult)
        else:
            raw_response_extract = llm_client.query(clinical_note, system_message_extract)
        final_result['thinking_process'].append(f"--- Step 1: Structured Extraction ---\nRaw response length: {len(str(raw_response_extract))} chars")
        parsed_extract = _safe_json_loads(raw_response_extract)
        if isinstance(parsed_extract, dict) and 'phenotypes' in parsed_extract:
            extracted_phenotypes = parsed_extract['phenotypes']
        elif isinstance(parsed_extract, list):
            for item in parsed_extract:
                if isinstance(item, dict) and 'HPO' in item and 'Phenotype' in item:
                    extracted_phenotypes.append(item)
    except Exception as e:
        logger.error(f"Structured extraction failed: {e}", exc_info=True)
        final_result['thinking_process'].append(f"Extraction error: {e}")
        final_result['thinking_process'] = "\n\n".join(final_result['thinking_process'])
        return final_result
    if not extracted_phenotypes:
        logger.warning("No phenotypes extracted in structured Step 1.")
        final_result['thinking_process'].append("No phenotypes extracted.")
        final_result['thinking_process'] = "\n\n".join(final_result['thinking_process'])
        return final_result
    logger.info(f"Structured extraction found {len(extracted_phenotypes)} phenotypes")
    final_result['thinking_process'].append(f"Extracted {len(extracted_phenotypes)} phenotypes successfully")
    logger.info("Step 2: Normalizing phenotypes using structured output")
    normalized_phenotypes = []
    for i, phenotype in enumerate(extracted_phenotypes):
        phenotype_desc = phenotype.get('Phenotype', '') if isinstance(phenotype, dict) else str(phenotype)
        if not phenotype_desc.strip(): continue
        try:
            if model_type == 'gemini':
                raw_normalized_response = llm_client.query(user_input=phenotype_desc, system_message=system_message_normalize, response_schema=PhenotypeNormalization)
            else:
                raw_normalized_response = llm_client.query(phenotype_desc, system_message_normalize)
            normalized_response = _safe_json_loads(raw_normalized_response)
            if (isinstance(normalized_response, dict) and normalized_response.get('hpo_term') and normalized_response.get('hpo_term') != 'none'):
                normalized_phenotypes.append({'original_term': normalized_response.get('original_term', phenotype_desc), 'hpo_term': normalized_response.get('hpo_term')})
                final_result['thinking_process'].append(f"--- Step 2 Normalize '{phenotype_desc}' ---\nNormalized to: {normalized_response.get('hpo_term')}")
            else:
                final_result['thinking_process'].append(f"--- Step 2 Normalize '{phenotype_desc}' ---\nCould not normalize (result: {normalized_response.get('hpo_term', 'none') if isinstance(normalized_response, dict) else 'none'})")
        except Exception as e:
            logger.error(f"Normalization failed for '{phenotype_desc}': {e}")
            final_result['thinking_process'].append(f"--- Step 2 Normalize '{phenotype_desc}' ---\nNormalization error: {e}")
            continue
    if not normalized_phenotypes:
        logger.warning("No phenotypes normalized in structured Step 2.")
        final_result['thinking_process'].append("No phenotypes successfully normalized.")
        final_result['thinking_process'] = "\n\n".join(final_result['thinking_process'])
        return final_result
    logger.info(f"Structured normalization completed: {len(normalized_phenotypes)} phenotypes")
    logger.info("Step 3: Vector retrieval for normalized phenotypes")
    final_hpo_terms = []
    for normalized in normalized_phenotypes:
        hpo_term = normalized['hpo_term']
        original_term = normalized['original_term']
        try:
            query_vec = embed_query(hpo_term, embeddings_model)
            distances, indices = faiss_index.search(query_vec, 1)
            if indices.size > 0 and indices[0][0] >= 0:
                best_idx = indices[0][0]
                best_score = distances[0][0]
                if best_score > 0.8:
                    best_match = embedded_documents[best_idx]
                    final_hpo_terms.append({'phrase': original_term, 'normalized_term': hpo_term, 'hpo_id': best_match.get('hp_id'), 'hpo_description': best_match.get('info'), 'similarity_score': float(best_score)})
                    final_result['thinking_process'].append(f"--- Step 3 Retrieval for '{hpo_term}' ---\n✅ Match: {best_match.get('hp_id')} - {best_match.get('info')} (score: {best_score:.3f})")
                else:
                    logger.info(f"Similarity score {best_score:.3f} too low for '{hpo_term}' (threshold: 0.8)")
                    final_result['thinking_process'].append(f"--- Step 3 Retrieval for '{hpo_term}' ---\n❌ Score {best_score:.3f} below threshold 0.8")
            else:
                final_result['thinking_process'].append(f"--- Step 3 Retrieval for '{hpo_term}' ---\n❌ No vector match found")
        except Exception as e:
            logger.error(f"Vector retrieval failed for '{hpo_term}': {e}")
            final_result['thinking_process'].append(f"--- Step 3 Retrieval for '{hpo_term}' ---\n❌ Retrieval error: {e}")
    final_result['hpo_terms'] = final_hpo_terms
    final_result['thinking_process'] = "\n\n".join(final_result['thinking_process'])
    cache.set(clinical_note, PIPELINE_KEY, final_result)
    logger.info(f"Cached structured result with {len(final_hpo_terms)} HPO terms")
    return final_result

def extract_thinking_from_content(content: str) -> tuple:
    if not content: return "", ""
    try:
        data = json.loads(content)
        if "choices" in data and data["choices"]:
            message_content = data["choices"][0].get("message", {}).get("content", "")
            if message_content: content = message_content
    except (json.JSONDecodeError, TypeError): pass
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, content, re.DOTALL)
    if matches:
        thinking_process = matches[0].strip()
        cleaned_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
        return thinking_process, cleaned_content
    else:
        return "", content

# --- [修改 2/4]: 修改此函數，使其接收 config 字典 ---
def check_and_initialize_llm(config: Dict[str, Any]):
    """Initialize LLM client from a config dictionary"""
    global model_type

    model_type = config.get("model_type")
    
    # 從傳入的 config 字典中準備參數
    common_args = {
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
        "model_name": config.get("model_name"),
        "temperature": config.get("temperature", 0.2),
        "max_tokens_per_day": config.get("max_tokens_per_day", -1),
        "max_queries_per_minute": config.get("max_queries_per_minute", 60),
        "max_tokens_per_minute": config.get("max_tokens_per_minute", 4000000),
        "think": config.get("think", False)
    }
    
    if model_type == "openai":
        del common_args['think']
        return LangchainOpenAIClient(**common_args)
    elif model_type == "gemini":
        return LangchainGeminiClient(**common_args)
    elif model_type == "ollama":
        return LangchainOllamaClient(**common_args)
    elif model_type == "vllm":
        return LangchainVLLMClient(**common_args)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

# ======================= Startup/Shutdown =======================
# --- [修改 3/4]: 重構 lifespan 以載入所有設定 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, embeddings_model, faiss_index, embedded_documents
    global system_message_extract, system_message_normalize
    global app_config, embedding_model_name  # 宣告要修改全域變數
    global meta_path, vec_path
    global system_prompt_path
    
    logger.info("Starting HPO Extraction API...")
    
    try:
        # 1. 載入 system prompts
        with open(system_prompt_path, "r") as f:
            prompts = json.load(f)
        system_message_extract = prompts.get("system_message_extract", "")
        system_message_normalize = prompts.get("system_message_normalize", "")
        
        if not all([system_message_extract, system_message_normalize]):
            raise ValueError("system_message_extract or system_message_normalize are missing from system_prompts.json")

        # 2. 載入 .flag 設定檔並存到全域變數
        if not os.path.exists(FLAG_FILE):
            raise FileNotFoundError(f"Flag file '{FLAG_FILE}' not found.")
        with open(FLAG_FILE, "r") as f:
            app_config = json.load(f)

        # 3. 初始化 LLM Client (傳入已載入的 config)
        llm_client = check_and_initialize_llm(app_config)
        
        # 4. 初始化 Embedding Model 並儲存名稱
        embeddings_model = SentenceTransformer(embedding_model_name)
        
        # 5. 載入向量資料庫
        docs, emb_matrix = load_vector_db(meta_path, vec_path)
        embedded_documents = docs
        faiss_index = create_faiss_index(emb_matrix)
        
        logger.info(f"API initialized with model type: '{app_config.get('model_type')}' and cache size: {cache.size()}")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    cache._save_cache()
    logger.info("Shutting down...")

# ======================= Pydantic Models =======================
# ... (此區塊程式碼不變)
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
app = FastAPI(
    title="Modified HPO Extraction API",
    description="HPO term extraction with Extract -> Normalize -> Retrieve pipeline.",
    version="2.0",
    lifespan=lifespan
)
# ... (middleware 不變)
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
        "service": "Modified HPO Extraction API",
        "version": "2.0",
        "cache_size": cache.size(),
        "pipeline": "Extract -> Normalize -> Retrieve"
    }

# --- [修改 4/4]: 新增 /config 端點 ---
@app.get("/config")
async def get_configuration():
    """
    Display the current application configuration, excluding sensitive information.
    """
    # 複製一份設定，避免修改到原始的 global config
    safe_config = app_config.copy()
    
    # 移除 api_key，確保不會洩漏
    safe_config.pop("api_key", None)
    
    # 加入 embedding model 資訊
    safe_config["embedding_model"] = embedding_model_name
    
    return safe_config

@app.get("/health")
async def health():
# ... (之後的所有端點和主程式碼都不變)
    return {
        "status": "healthy",
        "cache_size": cache.size(),
        "timestamp": time.time()
    }
@app.post("/annotate", response_model=AnnotateResponse)
async def annotate(request: AnnotateRequest):
    try:
        start_time = time.time()
        was_cached = bool(cache.get(clean_note(request.text), "hpo_pipeline_v2.1_structured_output"))
        hpo_result = extract_hpo_terms_structured(request.text)
        hpo_terms = hpo_result.get('hpo_terms', [])
        thinking_process = hpo_result.get('thinking_process', '')
        term_parts = []
        id_parts = []
        seen_ids = set()
        for term in hpo_terms:
            phrase = term.get('phrase', '')
            hpo_id = term.get('hpo_id', '')
            if hpo_id and hpo_id.startswith("HP:") and hpo_id not in seen_ids:
                term_parts.append(f"{phrase} ({hpo_id})")
                id_parts.append(hpo_id)
                seen_ids.add(hpo_id)
        return AnnotateResponse(text=request.text, hpo_terms=";".join(term_parts), hpo_ids=";".join(id_parts), processing_time=time.time() - start_time, cached=was_cached, thinking_process=thinking_process)
    except Exception as e:
        logger.error(f"Annotation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
recent_errors = []
def log_error_text(text: str, error: str, index: int = None):
    error_info = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'index': index, 'error': str(error), 'text_preview': text[:200] if text else "None", 'text_length': len(text) if text else 0, 'text_hash': hashlib.md5(text.encode() if text else b'').hexdigest()[:8]}
    recent_errors.append(error_info)
    if len(recent_errors) > 20:
        recent_errors.pop(0)
def save_problematic_text(text: str, error: str, index: int = None):
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
    try:
        start_time = time.time()
        results = []
        cached_count = 0
        total_texts = len(request.texts)
        logger.info(f"Starting batch processing of {total_texts} texts")
        PIPELINE_KEY = "hpo_pipeline_v2.1_structured_output"
        for i, text in enumerate(request.texts):
            try:
                text_length = len(text) if text else 0
                logger.info(f"Processing text {i+1}/{total_texts} (length: {text_length} chars)")
                if text and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Text preview: '{text[:100]}...'")
                if not text or not text.strip():
                    logger.warning(f"Empty text at index {i+1}, skipping")
                    results.append(AnnotateResponse(text=text or "", hpo_terms="", hpo_ids="", processing_time=0, cached=False, thinking_process=""))
                    continue
                cleaned_text = clean_note(text)
                was_cached = bool(cache.get(cleaned_text, PIPELINE_KEY))
                if not was_cached and llm_client and hasattr(llm_client, 'max_queries_per_minute'):
                    if llm_client.max_queries_per_minute > 0:
                        delay_time = 60.0 / llm_client.max_queries_per_minute
                        time.sleep(delay_time)
                text_start = time.time()
                if was_cached:
                    cached_count += 1
                hpo_result = extract_hpo_terms_structured(text)
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
                results.append(AnnotateResponse(text=text, hpo_terms=";".join(term_parts), hpo_ids=";".join(id_parts), processing_time=time.time() - text_start, cached=was_cached, thinking_process=thinking_process))
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
                results.append(AnnotateResponse(text=text or "", hpo_terms="ERROR: Processing failed", hpo_ids="ERROR", processing_time=0, cached=False, thinking_process=f"Error: {error_msg}"))
                continue
        total_processing_time = time.time() - start_time
        logger.info(f"Batch completed: {len(results)} results in {total_processing_time:.2f}s")
        return BatchResponse(results=results, total_time=total_processing_time, cached_count=cached_count, count=len(results), total_processing_time=total_processing_time)
    except Exception as e:
        logger.error(f"Batch processing completely failed: {e}", exc_info=True)
        cache._save_cache()
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
@app.get("/debug/recent-errors")
async def get_recent_errors():
    return {"total_recent_errors": len(recent_errors), "errors": recent_errors}
@app.post("/cache/clear")
async def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}
@app.get("/cache/stats")
async def cache_stats():
    return {"cache_size": cache.size(), "cache_file_exists": os.path.exists(cache.cache_file)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8555, reload=False)