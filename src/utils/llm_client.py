"""Unified LLM client supporting multiple providers."""

import json
import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI


class LLMClient:
    """
    Unified client for interacting with various LLM providers.

    Supported providers:
    - openai: OpenAI API (GPT-4, GPT-4o, etc.)
    - together: Together AI (Llama, Mistral, etc.)
    - anthropic: Anthropic API (Claude)
    """

    def __init__(
            self,
            model: str,
            provider: str = "together",
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize LLM client.

        Args:
            model: Model identifier (e.g., "meta-llama/Llama-3.1-70B-Instruct")
            provider: API provider ("openai", "together", "anthropic")
            api_key: API key (defaults to environment variable)
            base_url: Custom base URL for the API
            **kwargs: Additional configuration options
        """
        self.model = model
        self.provider = provider
        self.kwargs = kwargs

        # Set up API key and base URL based on provider
        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            base_url = base_url or "https://api.openai.com/v1"
        elif provider == "together":
            api_key = api_key or os.getenv("TOGETHER_API_KEY")
            base_url = base_url or "https://api.together.xyz/v1"
        elif provider == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            base_url = base_url or "https://api.anthropic.com/v1"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not api_key:
            raise ValueError(f"API key not found for provider: {provider}")

        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 256,
            top_p: float = 0.95,
            stop: Optional[List[str]] = None,
            json_mode: bool = False,
            **kwargs
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            json_mode: If True, request JSON output format
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            json_mode=json_mode,
            **kwargs
        )

    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 256,
            top_p: float = 0.95,
            stop: Optional[List[str]] = None,
            json_mode: bool = False,
            **kwargs
    ) -> str:
        """
        Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            json_mode: If True, request JSON output format
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        if stop:
            request_kwargs["stop"] = stop

        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        # Add any additional kwargs
        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)

        return response.choices[0].message.content

    def generate_json(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.

        Args:
            prompt: User prompt requesting JSON output
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default 0.0 for consistency)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Parsed JSON dictionary
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            **kwargs
        )

        # Parse JSON response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())


def create_client(
        model: str,
        provider: Optional[str] = None,
        **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        model: Model identifier
        provider: API provider (auto-detected if not provided)
        **kwargs: Additional configuration

    Returns:
        Configured LLMClient instance
    """
    # Auto-detect provider from model name
    if provider is None:
        if model.startswith("gpt-") or model.startswith("o1"):
            provider = "openai"
        elif model.startswith("claude"):
            provider = "anthropic"
        else:
            provider = "together"

    return LLMClient(model=model, provider=provider, **kwargs)
