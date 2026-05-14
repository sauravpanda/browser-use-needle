from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HF_REPO = "Cactus-Compute/needle"
DEFAULT_CHECKPOINT = "needle.pkl"


class NeedleOutputError(ValueError):
	"""Raised when Needle returns output that cannot be parsed as tool calls."""


def resolve_checkpoint(checkpoint: str | os.PathLike[str] | None = None, *, force_download: bool = False) -> str:
	"""Resolve a local checkpoint path or download a model file from Hugging Face."""
	if checkpoint:
		path = Path(checkpoint).expanduser()
		if path.exists():
			return str(path)
		filename = path.name
	else:
		filename = DEFAULT_CHECKPOINT

	from huggingface_hub import hf_hub_download

	local_dir = "checkpoints"
	os.makedirs(local_dir, exist_ok=True)
	return hf_hub_download(
		repo_id=DEFAULT_HF_REPO,
		filename=filename,
		repo_type="model",
		local_dir=local_dir,
		force_download=force_download,
	)


def normalize_tools_input(tools: list[dict[str, Any]] | str) -> str:
	if isinstance(tools, str):
		parsed = json.loads(tools)
	else:
		parsed = tools
	if not isinstance(parsed, list):
		raise TypeError("Needle tools must be a JSON array or a list of tool dictionaries.")
	return json.dumps(parsed, separators=(",", ":"))


@dataclass
class NeedleRuntime:
	"""Lazy loader for the Needle JAX model."""

	checkpoint: str | None = None
	max_gen_len: int = 512
	constrained: bool = True
	normalize: bool = True
	force_download: bool = False

	_model: Any = None
	_params: Any = None
	_tokenizer: Any = None
	_checkpoint_path: str | None = None

	def load(self) -> None:
		if self._model is not None:
			return

		from needle import SimpleAttentionNetwork, get_tokenizer, load_checkpoint

		self._checkpoint_path = resolve_checkpoint(self.checkpoint, force_download=self.force_download)
		self._params, config = load_checkpoint(self._checkpoint_path)
		self._model = SimpleAttentionNetwork(config)
		self._tokenizer = get_tokenizer()

	@property
	def checkpoint_path(self) -> str | None:
		return self._checkpoint_path

	def generate_text(self, query: str, tools: list[dict[str, Any]] | str) -> str:
		from needle import generate

		self.load()
		tools_json = normalize_tools_input(tools)
		result = generate(
			self._model,
			self._params,
			self._tokenizer,
			query=query,
			tools=tools_json,
			max_gen_len=self.max_gen_len,
			stream=False,
			normalize=self.normalize,
			constrained=self.constrained,
		)
		return result.strip()

	def generate_calls(self, query: str, tools: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
		raw = self.generate_text(query, tools)
		try:
			parsed = json.loads(raw)
		except json.JSONDecodeError as exc:
			raise NeedleOutputError(f"Needle returned invalid JSON: {raw}") from exc

		if isinstance(parsed, dict):
			parsed = [parsed]
		if not isinstance(parsed, list):
			raise NeedleOutputError(f"Needle returned a non-list tool call payload: {raw}")

		calls: list[dict[str, Any]] = []
		for item in parsed:
			if not isinstance(item, dict):
				continue
			name = item.get("name")
			arguments = item.get("arguments", {})
			if not isinstance(name, str) or not name:
				continue
			if not isinstance(arguments, dict):
				arguments = {}
			calls.append({"name": name, "arguments": arguments})
		return calls
