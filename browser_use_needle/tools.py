from __future__ import annotations

import json
from typing import Any


def browser_use_rust_tools() -> list[dict[str, Any]]:
	return [
		{
			"name": "click",
			"description": "Click an absolute screen coordinate.",
			"parameters": {
				"x": {"type": "integer", "description": "Horizontal coordinate from the left edge of the screen."},
				"y": {"type": "integer", "description": "Vertical coordinate from the top edge of the screen."},
			},
		},
		{
			"name": "type",
			"description": "Type text into the currently focused field.",
			"parameters": {
				"text": {"type": "string", "description": "Text to type."},
			},
		},
		{
			"name": "scroll",
			"description": "Scroll the screen up or down.",
			"parameters": {
				"direction": {"type": "string", "description": "Either up or down."},
				"amount": {"type": "integer", "description": "Scroll amount in pixels."},
			},
		},
		{
			"name": "hotkey",
			"description": "Press a keyboard shortcut such as cmd+c, cmd+shift+4, or alt+tab.",
			"parameters": {
				"keys": {"type": "string", "description": "Shortcut keys joined by plus signs."},
			},
		},
		{
			"name": "done",
			"description": "Finish the task and report the result.",
			"parameters": {
				"message": {"type": "string", "description": "Final message."},
			},
		},
	]


def _compact_property_schema(schema: dict[str, Any]) -> dict[str, Any]:
	out: dict[str, Any] = {}
	for key in ("type", "description", "default", "enum", "minimum", "maximum", "minLength", "maxLength"):
		if key in schema:
			out[key] = schema[key]

	if "type" not in out and "anyOf" in schema:
		for option in schema["anyOf"]:
			if isinstance(option, dict) and option.get("type") != "null":
				if "type" in option:
					out["type"] = option["type"]
				break

	if "items" in schema and isinstance(schema["items"], dict):
		out["items"] = _compact_property_schema(schema["items"])

	if not out:
		out["type"] = "string"
	return out


def pydantic_model_to_flat_parameters(model: type[Any]) -> dict[str, Any]:
	schema = model.model_json_schema()
	properties = schema.get("properties", {})
	if not isinstance(properties, dict):
		return {}
	return {name: _compact_property_schema(prop) for name, prop in properties.items() if isinstance(prop, dict)}


CORE_BROWSER_USE_ACTIONS = [
	"done",
	"search",
	"navigate",
	"go_back",
	"wait",
	"click",
	"input",
	"scroll",
	"send_keys",
]


def browser_use_python_tools(
	exclude_actions: list[str] | None = None,
	include_actions: list[str] | None = None,
) -> list[dict[str, Any]]:
	from browser_use.tools.service import Tools

	tools = Tools(exclude_actions=exclude_actions)
	return [
		{
			"name": action.name,
			"description": action.description or action.name,
			"parameters": pydantic_model_to_flat_parameters(action.param_model),
		}
		for action in tools.registry.registry.actions.values()
		if include_actions is None or action.name in include_actions
	]


def tools_for_preset(preset: str) -> list[dict[str, Any]]:
	if preset == "browser-use-rust":
		return browser_use_rust_tools()
	if preset == "browser-use-python-core":
		return browser_use_python_tools(include_actions=CORE_BROWSER_USE_ACTIONS)
	if preset == "browser-use-python":
		return browser_use_python_tools()
	raise ValueError(f"Unknown preset: {preset}")


def tools_json(tools: list[dict[str, Any]], *, pretty: bool = False) -> str:
	if pretty:
		return json.dumps(tools, indent=2, sort_keys=True)
	return json.dumps(tools, separators=(",", ":"))


def calls_to_browser_use_actions(calls: list[dict[str, Any]], max_actions: int = 1) -> list[dict[str, Any]]:
	actions: list[dict[str, Any]] = []
	for call in calls[:max_actions]:
		name = call.get("name")
		args = call.get("arguments", {})
		if isinstance(name, str) and isinstance(args, dict):
			actions.append({name: args})
	return actions


def rust_action_line(calls: list[dict[str, Any]]) -> str:
	if not calls:
		return "ACTION: done Needle returned no tool call"

	call = calls[0]
	name = call.get("name")
	args = call.get("arguments", {})
	if not isinstance(args, dict):
		args = {}

	if name == "click":
		return f"ACTION: click {int(args.get('x', 0))} {int(args.get('y', 0))}"
	if name == "type":
		return f"ACTION: type {str(args.get('text', ''))}"
	if name == "scroll":
		return f"ACTION: scroll {str(args.get('direction', 'down'))} {int(args.get('amount', 500))}"
	if name == "hotkey":
		return f"ACTION: hotkey {str(args.get('keys', ''))}"
	if name == "done":
		return f"ACTION: done {str(args.get('message', args.get('text', 'Done')))}"
	return f"ACTION: done Unsupported Needle tool call: {name}"
