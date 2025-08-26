# -*- coding: utf-8 -*-
import time
import requests
import tiktoken
import logging

logger = logging.getLogger(__name__)

class OpenAICompatibleClient:
    """
    LLM Client for OpenAI-compatible endpoints, like the one Google provides.
    Does NOT support safety settings in the payload.
    """
    def __init__(self, api_key, base_url, model_name, 
                 max_tokens_per_day=4000000, max_queries_per_minute=4000, 
                 max_tokens_per_minute=4000000, temperature=0.7):
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
        """Reset tokens_used_this_minute counter every minute"""
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            self.tokens_used_this_minute = 0
            self.last_minute_reset = current_time

    def query(self, user_input, system_message):
        tokens_ui = len(self.encoder.encode(user_input))
        tokens_sys = len(self.encoder.encode(system_message))
        estimated = tokens_ui + tokens_sys
        
        self._reset_minute_counter_if_needed()
        
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        
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
            self.total_tokens_used += result["usage"]["total_tokens"]
        else:
            self.total_tokens_used += estimated
            
        choices = result.get("choices") or []
        return choices[0].get("message", {}).get("content", "") if choices else ""
    

class GeminiNativeClient:
    """
    LLM Client for Google's native Gemini API endpoint.
    Supports safety settings.
    """
    def __init__(self, api_key, base_url, model_name, 
                 max_tokens_per_day=4000000, max_queries_per_minute=4000, 
                 max_tokens_per_minute=4000000, temperature=0.7):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name # Note: base_url already contains model for native API
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.temperature = temperature
        self.total_tokens_used = 0
        self.tokens_used_this_minute = 0
        self.last_minute_reset = time.time()
        # Native API does not use Authorization header, key is in URL
        self.headers = {"Content-Type": "application/json"}
        try:
            # Gemini models generally use cl100k_base
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            self.encoder = tiktoken.encoding_for_model("gpt-4") # fallback

    def _reset_minute_counter_if_needed(self):
        """Reset tokens_used_this_minute counter every minute"""
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            self.tokens_used_this_minute = 0
            self.last_minute_reset = current_time

    def query(self, user_input, system_message):
        tokens_ui = len(self.encoder.encode(user_input))
        tokens_sys = len(self.encoder.encode(system_message))
        estimated = tokens_ui + tokens_sys
        
        self._reset_minute_counter_if_needed()
        
        if self.max_tokens_per_day != -1 and self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Daily token limit exceeded.")
        
        if self.max_tokens_per_minute != -1 and self.tokens_used_this_minute + estimated > self.max_tokens_per_minute:
            raise Exception("Tokens per minute limit exceeded. Please wait.")
        
        time.sleep(60 / self.max_queries_per_minute)

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": f"{system_message}\n\n{user_input}"}]
            }],
            "generationConfig": {
                "temperature": self.temperature,
            },
            "safetySettings": safety_settings
        }
        
        try:
            url_with_key = f"{self.base_url}?key={self.api_key}"
            resp = requests.post(url_with_key, headers=self.headers, json=payload)
            resp.raise_for_status()
            result = resp.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed (Gemini Native): {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise Exception(f"LLM API call failed: {str(e)}")
        
        if "usageMetadata" in result and "totalTokenCount" in result["usageMetadata"]:
            self.total_tokens_used += result["usageMetadata"]["totalTokenCount"]
        else:
            self.total_tokens_used += estimated
            
        try:
            if "candidates" in result and result["candidates"]:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                return content
            elif result.get("promptFeedback", {}).get("blockReason"):
                 block_reason = result["promptFeedback"]["blockReason"]
                 error_message = f"Request was blocked by API for safety reasons: {block_reason}"
                 logger.error(error_message)
                 return f"Error: {error_message}"
            else:
                logger.warning(f"API returned no candidates and no block reason. Full response: {result}")
                return ""
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse LLM response: {e}. Full response: {result}")
            return ""