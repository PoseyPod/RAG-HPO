# -*- coding: utf-8 -*-
import time
import tiktoken
import logging
import json
import os
from pydantic import BaseModel

# --- LangChain Core Imports ---
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.exceptions import OutputParserException

# --- LangChain Model Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

# ==============================================================================
#  [重構] 1. Langchain-based OpenAI / OpenAI-Compatible Client
# ==============================================================================
class LangchainOpenAIClient:
    """
    LLM Client for OpenAI-compatible endpoints using LangChain.
    Replaces the original requests-based implementation.
    The public interface (__init__, query) remains the same.
    """
    def __init__(self, api_key, base_url, model_name,
                 max_tokens_per_day=4000000, max_queries_per_minute=4000,
                 max_tokens_per_minute=4000000, temperature=0.7, **kwargs):
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        # --- [變更]: 初始化 LangChain Client ---
        # 不再需要手動管理 headers，直接初始化 ChatOpenAI client
        self.client = ChatOpenAI(
            model=self.model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs  # 傳遞額外參數，如 max_tokens 等
        )

        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()

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

        # --- [不變]: 速率限制邏輯保持不變 ---
        self._reset_minute_counter_if_needed()
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        if self.max_queries_per_minute > 0:
            time.sleep(60 / self.max_queries_per_minute)

        # 建立 LangChain messages
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_input)
        ]

        try:
            # --- [變更]: 使用 LangChain .invoke() 取代 requests.post ---
            response = self.client.invoke(messages)

            # 從 LangChain 的 AIMessage 回應中獲取 token 用量
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = response.usage_metadata.get("total_tokens", 0)
                if tokens_used > 0:
                    self.total_tokens_used += tokens_used
                    self.tokens_used_this_minute += tokens_used
                else: # 如果回傳為 0，使用估算值
                    self.total_tokens_used += estimated
                    self.tokens_used_this_minute += estimated
            else: # 如果沒有 usage_metadata，回退到估算值
                self.total_tokens_used += estimated
                self.tokens_used_this_minute += estimated

            return response.content

        except Exception as e:
            logger.error(f"API request failed (Langchain OpenAI): {e}")
            raise Exception(f"LLM API call failed: {str(e)}")

# ==============================================================================
#  2. LangchainGeminiClient (保持不變，作為參考)
# ==============================================================================
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
        return "gemini-1.5" in model_lower

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

            if isinstance(response, AIMessage):
                try:
                    usage_metadata = response.response_metadata.get("usage_metadata")
                    if usage_metadata and "total_token_count" in usage_metadata:
                        tokens_used = usage_metadata["total_token_count"]
                        self.total_tokens_used += tokens_used
                        self.tokens_used_this_minute += tokens_used
                    else:
                        self.total_tokens_used += estimated
                        self.tokens_used_this_minute += estimated
                except Exception:
                    self.total_tokens_used += estimated
                    self.tokens_used_this_minute += estimated
            else:
                self.total_tokens_used += estimated
                self.tokens_used_this_minute += estimated

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
            raise Exception(f"LLM API call failed: {str(e)}")

# ==============================================================================
#  [重構] 3. Langchain-based vLLM Client
# ==============================================================================
class LangchainVLLMClient:
    """
    LLM Client for vLLM server using LangChain's OpenAI-compatible client.
    The public interface (__init__, query) remains the same.
    """
    def __init__(self, api_key=None, base_url="http://localhost:7111/v1/chat/completions",
                 model_name=None, max_tokens_per_day=-1,
                 max_queries_per_minute=120, max_tokens_per_minute=-1, temperature=0.7, **kwargs):
        self.model_name = model_name or "loaded-model"
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        # --- [變更]: 使用 ChatOpenAI 與 vLLM 的相容接口通訊 ---
        # vLLM 提供 OpenAI 相容的 API，所以我們可以直接使用 ChatOpenAI 客戶端
        self.client = ChatOpenAI(
            model=self.model_name,
            api_key=api_key or "vllm", # 保持原本的預設值
            base_url=base_url,
            temperature=temperature,
            max_tokens=2048, # 保持原本的預設值
            **kwargs
        )

        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()

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

        # --- [不變]: 速率限制邏輯保持不變 ---
        self._reset_minute_counter_if_needed()
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        if self.max_queries_per_minute > 0:
            time.sleep(60 / self.max_queries_per_minute)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_input)
        ]

        try:
            # --- [變更]: 使用 LangChain .invoke() 取代 requests.post ---
            response = self.client.invoke(messages)

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = response.usage_metadata.get("total_tokens", 0)
                if tokens_used > 0:
                    self.total_tokens_used += tokens_used
                    self.tokens_used_this_minute += tokens_used
                else:
                    self.total_tokens_used += estimated
                    self.tokens_used_this_minute += estimated
            else:
                self.total_tokens_used += estimated
                self.tokens_used_this_minute += estimated

            return response.content

        except Exception as e:
            logger.error(f"API request failed (Langchain vLLM): {e}")
            raise Exception(f"vLLM API call failed: {str(e)}")

# ==============================================================================
#  [重構] 4. Langchain-based Ollama Client
# ==============================================================================
class LangchainOllamaClient:
    """
    LLM Client for Ollama using LangChain.
    Replaces the original requests-based implementation.
    The public interface (__init__, query) remains the same.
    """
    def __init__(self, api_key=None, base_url="http://localhost:11434/v1/chat/completions",
                 model_name="llama3", max_tokens_per_day=-1, max_queries_per_minute=60,
                 max_tokens_per_minute=-1, temperature=0.7, think=False, **kwargs):
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.think = think

        # --- [變更]: 初始化 LangChain ChatOllama Client ---
        # 注意: ChatOllama 的 base_url 預期是 Ollama 服務的根 URL，而不是 completions 端點
        # 我們需要從 "http://.../v1/chat/completions" 中移除路徑
        ollama_root_url = base_url.split('/v1/')[0]

        self.client = ChatOllama(
            model=self.model_name,
            base_url=ollama_root_url,
            temperature=temperature,
            **kwargs
        )

        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()

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

        # --- [不變]: 速率限制邏輯保持不變 ---
        self._reset_minute_counter_if_needed()
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        if self.max_queries_per_minute > 0:
            time.sleep(60 / self.max_queries_per_minute)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_input)
        ]

        try:
            # --- [變更]: 使用 LangChain .invoke() 取代 requests.post ---
            model_runnable = self.client
            # 處理 think 模式，這是 Ollama 的一個非標準參數
            if self.think:
                # 使用 .bind() 將非標準參數傳遞給模型
                model_runnable = self.client.bind(options={"think": True})

            response = model_runnable.invoke(messages)

            # 從 Ollama 的回傳元數據中獲取 token 用量
            if hasattr(response, 'response_metadata') and response.response_metadata:
                prompt_tokens = response.response_metadata.get('prompt_eval_count', 0)
                completion_tokens = response.response_metadata.get('eval_count', 0)
                tokens_used = prompt_tokens + completion_tokens

                if tokens_used > 0:
                    self.total_tokens_used += tokens_used
                    self.tokens_used_this_minute += tokens_used
                else:
                    self.total_tokens_used += estimated
                    self.tokens_used_this_minute += estimated
            else:
                self.total_tokens_used += estimated
                self.tokens_used_this_minute += estimated

            # 根據 think 模式決定回傳內容
            if self.think:
                # 保持原始行為，回傳包含 thought 的完整 JSON
                # LangChain 會將原始回應放在 response_metadata 中
                return json.dumps(response.response_metadata)
            else:
                return response.content
        except Exception as e:
            logger.error(f"API request failed (Langchain Ollama): {e}")
            raise Exception(f"Ollama API call failed: {str(e)}")