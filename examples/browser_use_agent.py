from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from browser_use import Agent, Browser
from browser_use_needle import ChatNeedle


async def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", default="Go to https://example.com and report the page heading")
	parser.add_argument("--max-steps", type=int, default=4)
	parser.add_argument("--checkpoint")
	args = parser.parse_args()

	load_dotenv()

	agent = Agent(
		task=args.task,
		llm=ChatNeedle(checkpoint=args.checkpoint),
		browser=Browser(),
		use_vision=False,
		use_judge=False,
		enable_planning=False,
		max_actions_per_step=1,
	)
	await agent.run(max_steps=args.max_steps)


if __name__ == "__main__":
	asyncio.run(main())
