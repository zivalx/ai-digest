"""
Structured LLM calls through the local Claude Code CLI (``claude -p``).

Lets extraction run on the machine's signed-in Claude Code account instead of
an Anthropic API key: select it with ``LLM_PROVIDERS=claude_code``. Each call is
one non-interactive ``claude -p`` run with no tools, a replaced system prompt
and ``--json-schema`` structured output, validated into the given Pydantic model.

``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` are removed from the child's
environment so the CLI uses its own sign-in, not a key loaded from ``.env``.
"""
import asyncio
import json
import os
import shutil
import tempfile
from typing import Any, List, Optional, Type

from pydantic import BaseModel

PROVIDER = "claude_code"
DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT_SECONDS = 600
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class ClaudeCodeError(RuntimeError):
    """The CLI ran but returned an error (message is the CLI's own text)."""


def claude_code_available(cli: str = "claude") -> bool:
    return shutil.which(cli) is not None


def _split_messages(messages: List[Any]) -> tuple:
    system = "\n\n".join(m.content for m in messages if getattr(m, "type", "") == "system")
    user = "\n\n".join(m.content for m in messages if getattr(m, "type", "") != "system")
    return system, user


class ClaudeCodeStructured:
    """``ainvoke(messages) -> schema instance``, like ``with_structured_output``."""

    def __init__(self, schema: Type[BaseModel], model: Optional[str] = None,
                 cli: str = "claude", timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.schema = schema
        self.model = model or os.getenv("CLAUDE_CODE_MODEL", DEFAULT_MODEL)
        self.cli = cli
        self.timeout = timeout
        # Empty working dir: no project CLAUDE.md or local settings get picked up.
        self._cwd = tempfile.mkdtemp(prefix="claude-code-extract-")

    def command(self, system_prompt: str) -> List[str]:
        return [
            self.cli, "-p",
            "--model", self.model,
            "--system-prompt", system_prompt,
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "local",
            "--no-session-persistence",
            "--output-format", "json",
            "--json-schema", json.dumps(self.schema.model_json_schema()),
        ]

    async def ainvoke(self, messages: List[Any]) -> BaseModel:
        system, user = _split_messages(messages)
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        proc = await asyncio.create_subprocess_exec(
            *self.command(system), cwd=self._cwd, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(user.encode("utf-8")), self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise ClaudeCodeError(f"claude -p timed out after {self.timeout:.0f}s")
        return self.parse(out, err, proc.returncode)

    def parse(self, out: bytes, err: bytes, returncode: Optional[int]) -> BaseModel:
        try:
            payload = json.loads(out.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            detail = (err or out).decode("utf-8", "replace").strip()[:500]
            raise ClaudeCodeError(f"claude -p failed (exit {returncode}): {detail}")
        if payload.get("is_error") or payload.get("subtype") != "success":
            raise ClaudeCodeError(f"claude -p error: {str(payload.get('result'))[:500]}")
        structured = payload.get("structured_output")
        if structured is None:
            raise ClaudeCodeError("claude -p returned no structured_output")
        return self.schema.model_validate(structured)
