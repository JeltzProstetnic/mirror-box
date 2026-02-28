"""LLM backend — model loading and inference engine for the Mirror Box.

Wraps a HuggingFace Causal LLM (default: Mistral-7B INT4) and provides
two inference paths:

1. **Encoder path** (substrate/implicit models): Uses the first N transformer
   layers to produce state vectors. Designed for ~40 Hz so it must be cheap.
   Called via ``encode(input_ids)``.

2. **Generator path** (explicit models): Uses the full model for either
   hidden-state extraction or text generation. Runs at ~20 Hz.
   Called via ``generate_hidden(input_ids)`` or ``generate_text(prompt)``.

The LLMBackend does NOT manage LoRA adapters — that is ISM's responsibility
(in implicit.py). This class provides the base model that adapters attach to.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel

    from mirror_box.config import MirrorBoxConfig

logger = logging.getLogger(__name__)


class LLMBackend:
    """Model loading and inference engine for the Mirror Box.

    Loads a causal language model with optional quantization and provides
    encode/generate methods for the substrate and simulation layers.

    Args:
        config: MirrorBoxConfig with model, quantization, and device settings.
        model: Optional pre-loaded model (for testing with TinyMockLLM).
            If provided, skips HuggingFace loading entirely.
        tokenizer: Optional pre-loaded tokenizer (paired with custom model).
    """

    def __init__(
        self,
        config: MirrorBoxConfig,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._config = config
        self._device = config.hardware.resolve_device()

        if model is not None:
            # Testing path: use provided model directly
            self._model = model.to(self._device)
            self._tokenizer = tokenizer
            self._hidden_size = self._detect_hidden_size(model)
        else:
            # Production path: load from HuggingFace
            self._model, self._tokenizer = self._load_model()
            self._hidden_size = self._model.config.hidden_size

        # Persistent token buffer for continuous generation (working memory).
        # Accumulates all generated text — the model sees its full history
        # on each call. Cleared when nearing max_context_length (fatigue).
        self._token_buffer: list[int] | None = None
        self._kv_token_count: int = 0

        # Projection layer: bridge hidden_size -> state_dim if they differ
        self._dtype = config.hardware.resolve_dtype()
        if self._hidden_size != config.state_dim:
            self._projection: nn.Linear | None = nn.Linear(
                self._hidden_size, config.state_dim
            ).to(device=self._device, dtype=self._dtype)
            logger.info(
                "Projection layer created: %d -> %d",
                self._hidden_size,
                config.state_dim,
            )
        else:
            self._projection = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple[PreTrainedModel, AutoTokenizer]:
        """Load model and tokenizer from HuggingFace with quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self._config.model_name_or_path
        quant = self._config.quantization
        device = self._device

        logger.info("Loading model %s with quantization=%s on %s", model_id, quant, device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if quant == "nf4_bnb":
            model = self._load_nf4_bnb(model_id)
        elif quant == "int4_gptq":
            model = self._load_int4_gptq(model_id)
        else:
            # quantization="none" — full precision
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=self._config.hardware.resolve_dtype(),
                device_map="auto" if device.type == "cuda" else None,
                trust_remote_code=True,
            )
            if device.type != "cuda":
                model = model.to(device)

        model.eval()
        return model, tokenizer

    def _load_nf4_bnb(self, model_id: str) -> PreTrainedModel:
        """Load with bitsandbytes NF4 quantization (CUDA-only)."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "bitsandbytes quantization requires the 'bitsandbytes' package. "
                "Install it with: pip install bitsandbytes"
            ) from e

        if self._device.type != "cuda":
            logger.warning(
                "nf4_bnb quantization requires CUDA but device is %s. "
                "Falling back to full precision.",
                self._device,
            )
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=self._config.hardware.resolve_dtype(),
                trust_remote_code=True,
            )
            return model.to(self._device)

        from transformers import AutoModelForCausalLM

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self._config.hardware.resolve_dtype(),
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    def _load_int4_gptq(self, model_id: str) -> PreTrainedModel:
        """Load with AutoGPTQ INT4 quantization (CUDA-only)."""
        try:
            from transformers import GPTQConfig
        except ImportError as e:
            raise ImportError(
                "GPTQ quantization requires the 'auto-gptq' package. "
                "Install it with: pip install auto-gptq"
            ) from e

        if self._device.type != "cuda":
            logger.warning(
                "int4_gptq quantization requires CUDA but device is %s. "
                "Falling back to full precision.",
                self._device,
            )
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=self._config.hardware.resolve_dtype(),
                trust_remote_code=True,
            )
            return model.to(self._device)

        from transformers import AutoModelForCausalLM

        gptq_config = GPTQConfig(
            bits=4,
            disable_exllama=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=gptq_config,
            device_map="auto",
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # Hidden size detection (for custom models without .config)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_hidden_size(model: nn.Module) -> int:
        """Detect hidden size from a model that may not have a .config attribute.

        Probes the model with a dummy forward pass to determine output
        dimensionality. Works with TinyMockLLM and any model that accepts
        input_ids and returns hidden states.
        """
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        if hasattr(model, "hidden_size"):
            return model.hidden_size

        # Fallback: probe with dummy input
        model.eval()
        device = next(model.parameters()).device
        dummy = torch.tensor([[0]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(dummy, output_hidden_states=True)
        if hasattr(out, "hidden_states") and out.hidden_states:
            return out.hidden_states[0].shape[-1]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state.shape[-1]
        raise RuntimeError(
            "Cannot detect hidden_size: model output has neither "
            "hidden_states nor last_hidden_state."
        )

    # ------------------------------------------------------------------
    # Encoder path (substrate, ~40 Hz)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the first N layers, mean-pooled to (state_dim,).

        This is the cheap path for substrate encoding at ~40 Hz. Only the
        first ``encoder_layer_index`` layers execute, then hidden states are
        mean-pooled across the sequence dimension and optionally projected
        to ``state_dim``.

        Args:
            input_ids: Token IDs. Shape: ``(batch, seq_len)`` or ``(seq_len,)``.
                If 1-D, a batch dimension is added automatically.

        Returns:
            State vector. Shape: ``(state_dim,)``.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self._device)

        out = self._model(input_ids, output_hidden_states=True)
        # Select the layer at encoder_layer_index (clamped to available layers)
        hidden_states = out.hidden_states
        layer_idx = min(self._config.encoder_layer_index, len(hidden_states) - 1)
        h = hidden_states[layer_idx]  # (batch, seq, hidden_size)

        # Mean-pool across sequence dimension, squeeze batch
        pooled = h.mean(dim=1).squeeze(0)  # (hidden_size,)

        if self._projection is not None:
            pooled = self._projection(pooled)

        return pooled

    # ------------------------------------------------------------------
    # Generator path (explicit, ~20 Hz)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass, return last hidden state mean-pooled to (state_dim,).

        Uses all layers (not truncated). More expensive than ``encode()``
        but produces richer representations for the explicit model layer.

        Args:
            input_ids: Token IDs. Shape: ``(batch, seq_len)`` or ``(seq_len,)``.

        Returns:
            State vector. Shape: ``(state_dim,)``.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self._device)

        out = self._model(input_ids, output_hidden_states=True)
        # Use the final hidden state (last element of hidden_states tuple)
        h = out.hidden_states[-1]  # (batch, seq, hidden_size)
        pooled = h.mean(dim=1).squeeze(0)  # (hidden_size,)

        if self._projection is not None:
            pooled = self._projection(pooled)

        return pooled

    def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Full text generation with sampling.

        Uses the tokenizer to encode the prompt, runs autoregressive
        generation with temperature sampling, and decodes the output.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate. Defaults to
                ``config.generation_max_tokens``.

        Returns:
            Generated text (excluding the prompt).

        Raises:
            RuntimeError: If no tokenizer is available (e.g., testing
                with TinyMockLLM without a tokenizer).
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "generate_text() requires a tokenizer. When using a custom "
                "model (e.g., TinyMockLLM), provide a tokenizer or use "
                "encode()/generate_hidden() instead."
            )

        max_new = max_tokens or self._config.generation_max_tokens

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_context_length,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                temperature=self._config.generation_temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the new tokens (exclude prompt)
        new_tokens = output_ids[0, input_ids.shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Persistent KV cache — working memory
    # ------------------------------------------------------------------

    def generate_continuation(
        self,
        text: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text continuing the persistent KV cache stream.

        Unlike generate_text() which starts fresh each call, this method
        accumulates all context into a growing token buffer. Each call
        appends new tokens and regenerates from the full sequence.
        The model "remembers" because it sees everything that came before.

        When the buffer nears max_context_length, it is cleared (the model
        "forgets" and starts fresh). Use kv_fill_ratio() to monitor.

        Args:
            text: New text to append to the stream before generating.
            max_tokens: Max new tokens to generate.

        Returns:
            Generated text (excluding the input).
        """
        if self._tokenizer is None:
            raise RuntimeError("generate_continuation() requires a tokenizer.")

        max_new = max_tokens or self._config.generation_max_tokens
        max_ctx = self._config.max_context_length

        # Tokenize the new fragment
        new_ids = self._tokenizer.encode(text, add_special_tokens=False)

        # Append to running buffer
        if self._token_buffer is None:
            # First call — add BOS
            bos = self._tokenizer.bos_token_id
            self._token_buffer = [bos] + new_ids if bos is not None else list(new_ids)
        else:
            self._token_buffer.extend(new_ids)

        # Check if buffer would overflow — if so, reset (fatigue/forget)
        if len(self._token_buffer) + max_new > max_ctx:
            self.clear_kv_cache()
            bos = self._tokenizer.bos_token_id
            self._token_buffer = [bos] + new_ids if bos is not None else list(new_ids)

        input_ids = torch.tensor(
            [self._token_buffer], dtype=torch.long, device=self._device,
        )

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_new,
                temperature=self._config.generation_temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                repetition_penalty=1.3,
            )

        # Extract only the newly generated tokens
        gen_tokens = output_ids[0, input_ids.shape[1]:]
        # Add generated tokens to the buffer (model will see them next time)
        self._token_buffer.extend(gen_tokens.tolist())
        # Update fill counter
        self._kv_token_count = len(self._token_buffer)

        return self._tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def kv_fill_ratio(self) -> float:
        """How full the KV cache is (0.0 = empty, 1.0 = full).

        Use this to detect "fatigue" — when the working memory is
        nearly full and the system needs sleep to consolidate.
        """
        if self._kv_token_count == 0:
            return 0.0
        return self._kv_token_count / self._config.max_context_length

    def clear_kv_cache(self) -> None:
        """Clear the persistent token buffer (forget working memory)."""
        self._token_buffer = None
        self._kv_token_count = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_tokenizer(self) -> Any:
        """Return the tokenizer (may be None for test models)."""
        return self._tokenizer

    @property
    def model(self) -> nn.Module:
        """The underlying model (for LoRA adapter attachment by ISM)."""
        return self._model

    @property
    def device(self) -> torch.device:
        """The device the model is placed on."""
        return self._device

    @property
    def hidden_size(self) -> int:
        """The model's native hidden dimensionality (before projection)."""
        return self._hidden_size

    @property
    def state_dim(self) -> int:
        """Output dimensionality after optional projection."""
        return self._config.state_dim

    @property
    def has_projection(self) -> bool:
        """Whether a projection layer is active (hidden_size != state_dim)."""
        return self._projection is not None
