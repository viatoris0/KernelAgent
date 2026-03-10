# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local Transformers provider implementation (no endpoint)."""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from .base import BaseProvider, LLMResponse

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class LocalTransformersProvider(BaseProvider):
    """Run local Hugging Face models directly via transformers."""

    def __init__(self):
        self._model_cache: dict[str, tuple[Any, Any]] = {}
        super().__init__()

    def _initialize_client(self) -> None:
        # No remote client needed; model/tokenizer are loaded lazily per model_name.
        self.client = object() if TRANSFORMERS_AVAILABLE else None

    @property
    def name(self) -> str:
        return "local_transformers"

    def is_available(self) -> bool:
        return TRANSFORMERS_AVAILABLE and self.client is not None

    def _resolve_model_name(self, model_name: str) -> str:
        # Allow explicit local directory override while keeping registry model IDs.
        return os.getenv("LOCAL_MODEL_PATH", model_name)

    def _get_model_and_tokenizer(self, model_name: str) -> tuple[Any, Any]:
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers/torch are not installed. "
                "Install them to use LocalTransformersProvider."
            )

        resolved = self._resolve_model_name(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            resolved, trust_remote_code=True, local_files_only=True
        )
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": "auto",
        }
        if has_accelerate:
            model_kwargs["device_map"] = "auto"

        def _build_load_error(err: Exception) -> RuntimeError:
            msg = str(err)
            base = (
                f"Failed to load local model '{model_name}' from '{resolved}' "
                f"with AutoModelForCausalLM. Underlying error: {msg}"
            )
            if "requires `accelerate`" in msg or "requires accelerate" in msg:
                return RuntimeError(base + " Install accelerate: pip install accelerate")
            if any(k in msg.lower() for k in ["vision", "image", "video", "processor"]):
                return RuntimeError(
                    base
                    + " This simple provider is text-only; VLMs may require a processor-specific path."
                )
            return RuntimeError(base)

        try:
            model = AutoModelForCausalLM.from_pretrained(resolved, **model_kwargs)
        except Exception as e:
            # Common fallback: device_map="auto" path without accelerate or with
            # unsupported environment. Retry without device_map.
            if has_accelerate:
                try:
                    fallback_kwargs = dict(model_kwargs)
                    fallback_kwargs.pop("device_map", None)
                    model = AutoModelForCausalLM.from_pretrained(
                        resolved, **fallback_kwargs
                    )
                except Exception:
                    raise _build_load_error(e) from e
            else:
                raise _build_load_error(e) from e

        # Without accelerate/device_map, explicitly move the model to one device.
        if not has_accelerate:
            if torch.cuda.is_available():
                model = model.to("cuda")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                model = model.to("xpu")

        if (
            tokenizer.pad_token_id is None
            and tokenizer.eos_token_id is not None
            and tokenizer.pad_token is None
        ):
            tokenizer.pad_token = tokenizer.eos_token

        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        self._model_cache[model_name] = (tokenizer, model)
        return tokenizer, model

    def _messages_to_prompt(self, tokenizer: Any, messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback formatting for plain tokenizers.
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Local transformers provider not available")

        tokenizer, model = self._get_model_and_tokenizer(model_name)
        prompt = self._messages_to_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        max_new_tokens = min(kwargs.get("max_tokens", 1024), 4096)
        temperature = kwargs.get("temperature", 0.7)
        do_sample = temperature > 0

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        completion_ids = output_ids[0][input_len:]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True)

        return LLMResponse(
            content=content.strip(),
            model=model_name,
            provider=self.name,
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        responses: list[LLMResponse] = []
        base_temp = kwargs.get("temperature", 0.7)
        for i in range(max(n, 1)):
            responses.append(
                self.get_response(
                    model_name,
                    messages,
                    temperature=min(base_temp + i * 0.1, 1.5),
                    max_tokens=kwargs.get("max_tokens", 1024),
                )
            )
        return responses
