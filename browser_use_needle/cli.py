from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from browser_use_needle.runtime import NeedleRuntime
from browser_use_needle.tools import browser_use_rust_tools, rust_action_line, tools_for_preset, tools_json


PRESETS = ("browser-use-rust", "browser-use-python-core", "browser-use-python")
DEMO_CASES = [
	{
		"name": "type-text",
		"query": "The focused text input is ready. Type hello world.",
		"expected": "type",
	},
	{
		"name": "click-coordinate",
		"query": "A submit button is centered at x=640 y=360. Click the submit button.",
		"expected": "click",
	},
	{
		"name": "scroll-page",
		"query": "Scroll down by 700 pixels.",
		"expected": "scroll",
	},
	{
		"name": "finish-task",
		"query": "The task is complete. Report that the task finished successfully.",
		"expected": "done",
	},
]


def _read_query(args: argparse.Namespace) -> str:
	if args.query:
		return args.query
	if not sys.stdin.isatty():
		return sys.stdin.read().strip()
	raise SystemExit("Provide --query or pipe a query on stdin.")


def _read_tools(args: argparse.Namespace) -> list[dict]:
	if args.tools_file:
		return json.loads(Path(args.tools_file).read_text())
	if args.tools_json:
		parsed = json.loads(args.tools_json)
		if not isinstance(parsed, list):
			raise SystemExit("--tools-json must be a JSON array.")
		return parsed
	return tools_for_preset(args.preset)


def cmd_doctor(_: argparse.Namespace) -> int:
	print(f"cwd: {Path.cwd()}")
	print(f"python: {sys.version.split()[0]} ({sys.executable})")
	for rel in ("../browser-use", "../browser-use-rust"):
		path = (Path.cwd() / rel).resolve()
		print(f"{rel}: {'found' if path.exists() else 'missing'} ({path})")

	import browser_use
	import needle

	print(f"browser_use: {Path(browser_use.__file__).resolve()}")
	print(f"needle: {Path(needle.__file__).resolve()}")
	print("ok")
	return 0


def cmd_tools(args: argparse.Namespace) -> int:
	print(tools_json(tools_for_preset(args.preset), pretty=args.pretty))
	return 0


def cmd_run(args: argparse.Namespace) -> int:
	load_dotenv()
	query = _read_query(args)
	tools = _read_tools(args)
	runtime = NeedleRuntime(
		checkpoint=args.checkpoint,
		max_gen_len=args.max_gen_len,
		constrained=not args.no_constrained,
		force_download=args.force_download,
	)

	if args.raw:
		print(runtime.generate_text(query, tools))
		return 0

	calls = runtime.generate_calls(query, tools)
	if args.rust_action:
		print(rust_action_line(calls))
	else:
		print(json.dumps(calls, indent=2 if args.pretty else None))
	return 0


def cmd_demo(args: argparse.Namespace) -> int:
	load_dotenv()
	runtime = NeedleRuntime(
		checkpoint=args.checkpoint,
		max_gen_len=args.max_gen_len,
		constrained=not args.no_constrained,
		force_download=args.force_download,
	)
	tools = browser_use_rust_tools()

	failures = 0
	for index, case in enumerate(DEMO_CASES, 1):
		calls = runtime.generate_calls(case["query"], tools)
		actual = calls[0]["name"] if calls else "<none>"
		ok = actual == case["expected"]
		if not ok:
			failures += 1

		print(f"{index}. {case['name']}: {'ok' if ok else 'mismatch'}")
		print(f"   query: {case['query']}")
		print(f"   expected: {case['expected']}")
		print(f"   needle: {json.dumps(calls, separators=(',', ':'))}")
		if args.rust_action:
			print(f"   action: {rust_action_line(calls)}")

	if failures:
		print(f"\n{failures} demo case(s) differed from the expected tool name.")
		return 1
	print("\nAll demo cases matched the expected tool names.")
	return 0


def cmd_playground(args: argparse.Namespace) -> int:
	command = ["needle", "playground", "--host", args.host, "--port", str(args.port)]
	if args.checkpoint:
		command.extend(["--checkpoint", args.checkpoint])
	return subprocess.call(command, env=os.environ.copy())


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="buneedle")
	sub = parser.add_subparsers(dest="command", required=True)

	doctor = sub.add_parser("doctor", help="Check imports and sibling repo paths.")
	doctor.set_defaults(func=cmd_doctor)

	tools = sub.add_parser("tools", help="Print a Needle tool schema preset.")
	tools.add_argument("--preset", choices=PRESETS, default="browser-use-rust")
	tools.add_argument("--pretty", action="store_true")
	tools.set_defaults(func=cmd_tools)

	run = sub.add_parser("run", help="Run one Needle tool-call generation.")
	run.add_argument("--query", help="Text query/state to send to Needle. If omitted, stdin is used.")
	run.add_argument("--preset", choices=PRESETS, default="browser-use-rust")
	run.add_argument("--tools-json", help="Override tools with a JSON array string.")
	run.add_argument("--tools-file", help="Override tools with a JSON array file.")
	run.add_argument("--checkpoint", help="Local checkpoint path or Hugging Face model filename.")
	run.add_argument("--max-gen-len", type=int, default=512)
	run.add_argument("--no-constrained", action="store_true")
	run.add_argument("--force-download", action="store_true")
	run.add_argument("--raw", action="store_true", help="Print raw Needle text instead of parsed JSON.")
	run.add_argument("--pretty", action="store_true")
	run.add_argument("--rust-action", action="store_true", help="Convert the first call to ACTION: ... for browser-use-rust.")
	run.set_defaults(func=cmd_run)

	demo = sub.add_parser("demo", help="Run a Needle-only function-calling demo suite.")
	demo.add_argument("--checkpoint", help="Local checkpoint path or Hugging Face model filename.")
	demo.add_argument("--max-gen-len", type=int, default=512)
	demo.add_argument("--no-constrained", action="store_true")
	demo.add_argument("--force-download", action="store_true")
	demo.add_argument("--rust-action", action="store_true", help="Also print ACTION: ... lines for browser-use-rust.")
	demo.set_defaults(func=cmd_demo)

	playground = sub.add_parser("playground", help="Launch Needle's bundled playground UI.")
	playground.add_argument("--host", default="127.0.0.1")
	playground.add_argument("--port", type=int, default=7860)
	playground.add_argument("--checkpoint")
	playground.set_defaults(func=cmd_playground)

	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	return args.func(args)


if __name__ == "__main__":
	raise SystemExit(main())
