"""ClaudeCodeStructured: CLI command shape, output parsing, env scrubbing, chain wiring."""
import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.services import claude_code_llm as ccl
from src.services.extraction_service import ExtractionService, PickList, is_fatal_llm_error


def _ok(structured):
    return json.dumps({"type": "result", "subtype": "success", "is_error": False,
                       "result": "", "structured_output": structured}).encode()


def test_command_disables_tools_and_passes_schema_and_system_prompt():
    cmd = ccl.ClaudeCodeStructured(PickList, model="sonnet").command("SYS")
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == PickList.model_json_schema()


def test_parse_success_validates_into_schema():
    llm = ccl.ClaudeCodeStructured(PickList)
    out = llm.parse(_ok({"picks": []}), b"", 0)
    assert isinstance(out, PickList) and out.picks == []


def test_parse_cli_error_is_fatal_when_not_logged_in():
    llm = ccl.ClaudeCodeStructured(PickList)
    payload = json.dumps({"subtype": "success", "is_error": True,
                          "result": "Not logged in · Please run /login"}).encode()
    with pytest.raises(ccl.ClaudeCodeError) as exc:
        llm.parse(payload, b"", 1)
    assert is_fatal_llm_error(exc.value)


def test_parse_garbage_output_raises():
    with pytest.raises(ccl.ClaudeCodeError, match="exit 2"):
        ccl.ClaudeCodeStructured(PickList).parse(b"oops", b"boom", 2)


@pytest.mark.asyncio
async def test_ainvoke_strips_api_key_and_sends_user_text_on_stdin(monkeypatch):
    seen = {}

    class FakeProc:
        returncode = 0

        async def communicate(self, data):
            seen["stdin"] = data.decode()
            return _ok({"picks": []}), b""

    async def fake_exec(*cmd, cwd, env, **kw):
        seen["cmd"], seen["env"] = cmd, env
        return FakeProc()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    monkeypatch.setattr(ccl.asyncio, "create_subprocess_exec", fake_exec)
    out = await ccl.ClaudeCodeStructured(PickList).ainvoke(
        [SystemMessage(content="SYS"), HumanMessage(content="hello")])
    assert out.picks == []
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert seen["stdin"] == "hello"
    assert seen["cmd"][seen["cmd"].index("--system-prompt") + 1] == "SYS"


def test_chain_uses_claude_code_without_api_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ccl, "claude_code_available", lambda cli="claude": True)
    chain = ExtractionService(llm_providers="claude_code,anthropic").structured_llm
    assert chain.runnable.name == "extraction_provider_claude_code"
    assert list(chain.fallbacks) == []
