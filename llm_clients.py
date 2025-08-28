# -*- coding: utf-8 -*-
import time
import requests
import tiktoken
import logging
import json
import os
from pydantic import BaseModel, TypeAdapter

from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.exceptions import OutputParserException

logger = logging.getLogger(__name__)

class OpenAICompatibleClient:
    # ... (這個類別保持不變)
    """
    LLM Client for OpenAI-compatible endpoints, like the one Google provides.
    Does NOT support safety settings in the payload.
    """
    def __init__(self, api_key, base_url, model_name, 
                 max_tokens_per_day=4000000, max_queries_per_minute=4000, 
                 max_tokens_per_minute=4000000, temperature=0.7, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.temperature = temperature
        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            self.encoder = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def _reset_minute_counter_if_needed(self):
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            self.tokens_used_this_minute = 0
            self.last_minute_reset = current_time

    def query(self, user_input, system_message, **kwargs):
        tokens_ui = len(self.encoder.encode(user_input))
        tokens_sys = len(self.encoder.encode(system_message))
        estimated = tokens_ui + tokens_sys
        
        self._reset_minute_counter_if_needed()
        
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        
        if self.max_queries_per_minute > 0:
            time.sleep(60 / self.max_queries_per_minute)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            "temperature": self.temperature,
        }
        
        try:
            resp = requests.post(self.base_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed (OpenAI-compatible): {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise Exception(f"LLM API call failed: {str(e)}")
        
        if "usage" in result and "total_tokens" in result["usage"]:
            tokens_used = result["usage"]["total_tokens"]
            self.total_tokens_used += tokens_used
            self.tokens_used_this_minute += tokens_used
        else:
            self.total_tokens_used += estimated
            self.tokens_used_this_minute += estimated
            
        choices = result.get("choices") or []
        return choices[0].get("message", {}).get("content", "") if choices else ""

class LangchainGeminiClient:
    """
    LLM Client for Google's Gemini API using the Langchain library.
    Compatible with the existing FastAPI interface.
    Supports structured output and safety settings control.
    """
    def __init__(self, api_key=None, base_url=None, model_name="gemini-1.5-flash", 
                 max_tokens_per_day=4000000, max_queries_per_minute=4000, 
                 max_tokens_per_minute=4000000, temperature=0.7, 
                 thinking_budget=None, include_thoughts=False, disable_safety=True,
                 think=False):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.error("No API key provided and GOOGLE_API_KEY/GEMINI_API_KEY environment variable not set")
            raise ValueError("API key not provided for LangchainGeminiClient")
            
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.temperature = temperature
        self.disable_safety = disable_safety
        
        self.think = think
        if think:
            self.thinking_budget = thinking_budget if thinking_budget is not None else -1
            self.include_thoughts = True
        else:
            self.thinking_budget = thinking_budget
            self.include_thoughts = include_thoughts
            
        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()
        
        safety_settings = {}
        if self.disable_safety:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            safety_settings=safety_settings,
            # --- [修正 1/2]: 移除已棄用的參數 ---
            # convert_system_message_to_human=True 
        )
        
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            self.encoder = tiktoken.encoding_for_model("gpt-4")

    def _reset_minute_counter_if_needed(self):
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            self.tokens_used_this_minute = 0
            self.last_minute_reset = current_time

    def _is_gemini_15_series(self):
        model_lower = self.model_name.lower()
        return "gemini-1.5" in model_lower or "gemini-2.5" in model_lower

    def _format_thinking_response(self, response):
        try:
            if not isinstance(response, AIMessage):
                 return str(response)

            raw_candidates = response.response_metadata.get("candidates", [])
            if not raw_candidates:
                return response.content

            candidate = raw_candidates[0]
            response_parts = []
            thought_parts = []
            
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if not hasattr(part, 'text') or not part.text:
                        continue
                    
                    if hasattr(part, 'thought') and part.thought:
                        thought_parts.append(f"<think>\n{part.text.strip()}\n</think>")
                    else:
                        response_parts.append(part.text)

            if not thought_parts and not response_parts:
                 return response.content
            
            if thought_parts and response_parts:
                return f"{thought_parts[0]}\n\n{''.join(response_parts)}"
            elif thought_parts:
                return ''.join(thought_parts)
            elif response_parts:
                return ''.join(response_parts)
            else:
                return ""
                
        except Exception as e:
            logger.warning(f"Could not format thinking response: {e}")
            return response.content if hasattr(response, 'content') else ""

    def query(self, user_input, system_message="", response_schema=None):
        tokens_ui = len(self.encoder.encode(user_input))
        tokens_sys = len(self.encoder.encode(system_message)) if system_message else 0
        estimated = tokens_ui + tokens_sys
        
        self._reset_minute_counter_if_needed()
        
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        
        if self.max_queries_per_minute > 0:
            time.sleep(60 / self.max_queries_per_minute)
        
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=user_input))

        try:
            model_runnable = self.client
            
            if response_schema:
                model_runnable = self.client.with_structured_output(schema=response_schema)
            
            elif (self._is_gemini_15_series() and 
                  (self.thinking_budget is not None or self.include_thoughts)):
                
                thinking_config = {}
                if self.thinking_budget is not None:
                    thinking_config["thinking_budget"] = self.thinking_budget
                if self.include_thoughts:
                    thinking_config["include_thoughts"] = True
                
                if thinking_config:
                    generation_config = {"thinking_config": thinking_config}
                    model_runnable = self.client.bind(generation_config=generation_config)

            response = model_runnable.invoke(messages)
            
            # --- [修正 2/2]: 智慧地處理 token 計數 ---
            # 檢查回應是否為 AIMessage，只有它才包含元數據
            if isinstance(response, AIMessage):
                try:
                    usage_metadata = response.response_metadata.get("usage_metadata")
                    if usage_metadata and "total_token_count" in usage_metadata:
                        tokens_used = usage_metadata["total_token_count"]
                        self.total_tokens_used += tokens_used
                        self.tokens_used_this_minute += tokens_used
                    else: # 如果元數據中沒有 token 資訊，回退到估算值
                        self.total_tokens_used += estimated
                        self.tokens_used_this_minute += estimated
                except Exception:
                    # 任何讀取錯誤都回退到估算值
                    self.total_tokens_used += estimated
                    self.tokens_used_this_minute += estimated
            else:
                # 如果是結構化輸出 (Pydantic 物件)，元數據不可用，直接使用估算值
                # 這裡不需要印出警告，因為這是預期行為
                self.total_tokens_used += estimated
                self.tokens_used_this_minute += estimated

            # Format the output
            if response_schema:
                return response.model_dump_json()
            elif self.include_thoughts:
                return self._format_thinking_response(response)
            else:
                return response.content
                
        except OutputParserException as e:
            logger.error(f"Failed to parse structured output (Langchain): {e}")
            raise Exception(f"LLM failed to generate valid JSON: {e}")
        except Exception as e:
            logger.error(f"API request failed (Langchain Gemini): {e}")
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["blocked", "safety", "harm"]):
                return f"Error: Request was blocked by API for safety reasons: {str(e)}"
            elif "quota" in error_str or "limit" in error_str:
                return f"Error: API quota/limit exceeded: {str(e)}"
            elif "authentication" in error_str or "api key" in error_str:
                return f"Error: Authentication failed: {str(e)}"
            else:
                raise Exception(f"LLM API call failed: {str(e)}")

    def query_structured(self, user_input, system_message, response_schema):
        return self.query(user_input, system_message, response_schema)

class vLLMClient:
    # ... (這個類別保持不變)
    """
    LLM Client for vLLM server using OpenAI-compatible endpoint.
    Supports distributed inference and high-throughput serving.
    Model is pre-loaded in the vLLM server, so model_name is just for reference.
    """
    def __init__(self, api_key=None, base_url="http://localhost:7111/v1/chat/completions", 
                 model_name=None, max_tokens_per_day=-1, 
                 max_queries_per_minute=120, max_tokens_per_minute=-1, temperature=0.7, **kwargs):
        self.api_key = api_key or "vllm"
        self.base_url = base_url
        self.model_name = model_name or "loaded-model"
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.temperature = temperature
        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            logger.warning("Could not load tiktoken encoder, using approximation")
            self.encoder = None

    def _estimate_tokens(self, text):
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            return len(text) // 4

    def _reset_minute_counter_if_needed(self):
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            self.tokens_used_this_minute = 0
            self.last_minute_reset = current_time

    def query(self, user_input, system_message, **kwargs):
        tokens_ui = self._estimate_tokens(user_input)
        tokens_sys = self._estimate_tokens(system_message)
        estimated = tokens_ui + tokens_sys
        
        self._reset_minute_counter_if_needed()
        
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        
        if self.max_queries_per_minute > 0:
            time.sleep(60 / self.max_queries_per_minute)

        model_to_use = self.model_name
        if not model_to_use or model_to_use == "loaded-model":
            model_to_use = "default"
        
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            "temperature": self.temperature,
            "max_tokens": 2048,
        }
        
        try:
            resp = requests.post(self.base_url, headers=self.headers, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed (vLLM): {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise Exception(f"vLLM API call failed: {str(e)}")
        
        if "usage" in result and "total_tokens" in result["usage"]:
            tokens_used = result["usage"]["total_tokens"]
            self.total_tokens_used += tokens_used
            self.tokens_used_this_minute += tokens_used
        else:
            self.total_tokens_used += estimated
            self.tokens_used_this_minute += estimated
            
        try:
            choices = result.get("choices") or []
            if choices:
                return choices[0].get("message", {}).get("content", "")
            else:
                logger.warning(f"No choices in vLLM response: {result}")
                return ""
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to parse vLLM response: {e}. Full response: {result}")
            return ""

class OllamaClient:
    # ... (這個類別保持不變)
    """
    LLM Client for Ollama using OpenAI-compatible endpoint.
    No API key required, supports local models via OpenAI format.
    Supports thinking mode for compatible models like DeepSeek-R1.
    """
    def __init__(self, api_key=None, base_url="http://localhost:11434/v1/chat/completions", 
                 model_name="llama3", max_tokens_per_day=-1, max_queries_per_minute=60, 
                 max_tokens_per_minute=-1, temperature=0.7, think=False, **kwargs):
        self.api_key = api_key or "ollama"
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.temperature = temperature
        self.think = think
        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            logger.warning("Could not load tiktoken encoder, using approximation")
            self.encoder = None

    def _estimate_tokens(self, text):
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            return len(text) // 4

    def _reset_minute_counter_if_needed(self):
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            self.tokens_used_this_minute = 0
            self.last_minute_reset = current_time

    def query(self, user_input, system_message, **kwargs):
        tokens_ui = self._estimate_tokens(user_input)
        tokens_sys = self._estimate_tokens(system_message)
        estimated = tokens_ui + tokens_sys
        
        self._reset_minute_counter_if_needed()
        
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        
        if self.max_queries_per_minute > 0:
            time.sleep(60 / self.max_queries_per_minute)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            "temperature": self.temperature,
            "stream": False
        }
        
        if self.think:
            payload["think"] = True
        
        try:
            resp = requests.post(self.base_url, headers=self.headers, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed (Ollama): {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise Exception(f"Ollama API call failed: {str(e)}")
        
        if "usage" in result and "total_tokens" in result["usage"]:
            tokens_used = result["usage"]["total_tokens"]
            self.total_tokens_used += tokens_used
            self.tokens_used_this_minute += tokens_used
        else:
            self.total_tokens_used += estimated
            self.tokens_used_this_minute += estimated
        
        if self.think:
            return json.dumps(result)
        else:
            try:
                choices = result.get("choices") or []
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                else:
                    logger.warning(f"No choices in Ollama response: {result}")
                    return ""
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to parse Ollama response: {e}. Full response: {result}")
                return ""