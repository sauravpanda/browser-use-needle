from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion

from browser_use_needle.runtime import NeedleOutputError, NeedleRuntime
from browser_use_needle.tools import CORE_BROWSER_USE_ACTIONS, browser_use_python_tools, calls_to_browser_use_actions


def _message_text(message: BaseMessage) -> str:
	if hasattr(message, "text"):
		return str(message.text)
	content = getattr(message, "content", "")
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts = []
		for part in content:
			if getattr(part, "type", None) == "text":
				parts.append(getattr(part, "text", ""))
		return "\n".join(parts)
	return ""


class ChatNeedle:
	"""Experimental Browser Use chat adapter backed by Needle tool-call generation."""

	model = "Cactus-Compute/needle"

	def __init__(
		self,
		checkpoint: str | None = None,
		max_gen_len: int = 512,
		max_query_chars: int = 12000,
		max_actions: int = 1,
		constrained: bool = True,
	):
		self.runtime = NeedleRuntime(checkpoint=checkpoint, max_gen_len=max_gen_len, constrained=constrained)
		self.max_query_chars = max_query_chars
		self.max_actions = max_actions
		self._tools = browser_use_python_tools(include_actions=CORE_BROWSER_USE_ACTIONS)

	@property
	def provider(self) -> str:
		return "needle"

	@property
	def name(self) -> str:
		return self.model

	@property
	def model_name(self) -> str:
		return self.model

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[BaseModel] | None = None,
		**_: Any,
	) -> ChatInvokeCompletion[Any]:
		query = self._build_query(messages)
		try:
			calls = await asyncio.to_thread(self.runtime.generate_calls, query, self._tools)
		except NeedleOutputError as exc:
			if output_format is None:
				return ChatInvokeCompletion(completion=json.dumps({"error": str(exc)}), usage=None)
			completion = output_format.model_validate(self._fallback_payload(str(exc)))
			return ChatInvokeCompletion(completion=completion, usage=None)

		if output_format is None:
			return ChatInvokeCompletion(completion=json.dumps(calls, separators=(",", ":")), usage=None)

		payload = self._agent_output_payload(calls)
		try:
			completion = output_format.model_validate(payload)
		except ValidationError:
			completion = output_format.model_validate(self._fallback_payload("Needle returned an action Browser Use could not validate."))

		return ChatInvokeCompletion(completion=completion, usage=None)

	def _build_query(self, messages: list[BaseMessage]) -> str:
		blocks = []
		for message in messages:
			role = getattr(message, "role", "message")
			text = _message_text(message).strip()
			if text:
				blocks.append(f"{role.upper()}:\n{text}")

		query = "\n\n".join(blocks)
		if len(query) > self.max_query_chars:
			query = query[-self.max_query_chars :]

		return query

	def _agent_output_payload(self, calls: list[dict[str, Any]]) -> dict[str, Any]:
		actions = calls_to_browser_use_actions(calls, max_actions=self.max_actions)
		if not actions:
			return self._fallback_payload("Needle returned no action.")
		return {
			"evaluation_previous_goal": "Needle selected the next tool call from the current Browser Use state.",
			"memory": "Needle is running as an experimental local tool-call selector.",
			"next_goal": "Execute the selected action and observe the result.",
			"action": actions,
		}

	def _fallback_payload(self, message: str) -> dict[str, Any]:
		return {
			"evaluation_previous_goal": message,
			"memory": message,
			"next_goal": "Stop because the local Needle adapter could not produce a valid next action.",
			"action": [{"done": {"text": message, "success": False}}],
		}
