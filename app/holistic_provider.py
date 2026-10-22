"""Isolated Codex image inference for the opt-in holistic candidate."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from .config import OPENAI_MODEL, OPENAI_REASONING_EFFORT, ROOT
from .engine import file_fingerprint
from .holistic_scoring import METHOD, PROFILE, PROMPT_VERSION, prompt, schema
from .provider import ProviderError

DISABLED_FEATURES = ("shell_tool", "apps", "plugins", "hooks", "multi_agent", "multi_agent_v2",
                     "skill_search", "skill_mcp_dependency_install", "browser_use", "computer_use",
                     "image_generation", "in_app_browser")


def identity() -> dict:
    executable = shutil.which("codex")
    if not executable:
        raise ProviderError("未找到Codex CLI，未使用其他引擎替代")
    version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    return {"provider": "codex_cli_holistic", "model": OPENAI_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT, "cli_version": version,
            "scoring_method": METHOD, "prompt_version": PROMPT_VERSION, "profile": PROFILE,
            "implementation": {name: file_fingerprint(ROOT / "app" / name) for name in
                               ("holistic_scoring.py", "holistic_provider.py", "images.py")},
            "preprocessing": "oriented-rgb-max1800-face-crop1024-jpeg90-v1",
            "disabled_features": list(DISABLED_FEATURES), "skip_host_skill_discovery": True}


def command(executable: str, directory: Path, phase: str, images: list[Path]) -> list[str]:
    args = [executable, "--no-daemon", "-a", "never", "exec", "--ignore-user-config",
            "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
            "--model", OPENAI_MODEL, "-c", f'model_reasoning_effort="{OPENAI_REASONING_EFFORT}"',
            "-c", "project_doc_max_bytes=0", "-c", 'web_search="disabled"',
            "--enable", "skip_host_skill_discovery"]
    for feature in DISABLED_FEATURES:
        args.extend(["--disable", feature])
    args += ["--output-schema", str(directory / "schema.json"),
             "--output-last-message", str(directory / "result.json"), "--json"]
    for path in images:
        args += ["--image", str(path.resolve())]
    return args + ["-"]


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def evaluate(phase: str, image: Path, face_crop: Path | None, cancelled: Callable[[], bool] = lambda: False,
             timeout: float = 180) -> tuple[dict, dict]:
    if phase not in ("overall", "explanations"):
        raise ValueError("未知整体评分阶段")
    if cancelled():
        raise ProviderError("用户取消试评")
    execution_identity = identity()
    executable = shutil.which("codex")
    images = [image] + ([face_crop] if face_crop else [])
    hashes = [file_fingerprint(p) for p in images]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="beautypick-inference-") as temp:
        directory = Path(temp)
        (directory / "schema.json").write_text(json.dumps(schema(phase)), encoding="utf-8")
        args = command(executable, directory, phase, images)
        # Events and diagnostics remain local; only final JSON is consumed as a score.
        with (directory / "events.jsonl").open("w+") as events, (directory / "stderr.txt").open("w+") as errors:
            process = subprocess.Popen(args, cwd=directory, stdin=subprocess.PIPE, stdout=events,
                                       stderr=errors, text=True, start_new_session=True)
            try:
                process.stdin.write(prompt(phase, face_crop is not None))
                process.stdin.close()
                while process.poll() is None:
                    if cancelled():
                        raise ProviderError("用户取消试评")
                    if time.monotonic() - started > timeout:
                        raise ProviderError(f"{phase}阶段超过{timeout:g}秒，未返回评分")
                    time.sleep(0.2)
                events.seek(0)
                event_list = []
                for line in events:
                    try:
                        event_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                for event in event_list:
                    item_type = event.get("item", {}).get("type", "")
                    if item_type in ("command_execution", "mcp_tool_call", "web_search", "tool_call"):
                        raise ProviderError("试评出现工具调用，结果已拒绝")
                if process.returncode != 0:
                    # CLI failures may contain account details; expose only API error messages.
                    error_events = [e.get("message") or e.get("error", {}).get("message", "")
                                    for e in event_list if e.get("type") in ("error", "turn.failed")]
                    safe_error = next((e for e in error_events if e), "Codex CLI执行失败")
                    raise ProviderError(f"{phase}失败：{safe_error[:250]}")
                output = directory / "result.json"
                if not output.exists():
                    raise ProviderError("本次试评没有返回结构化结果")
                raw = json.loads(output.read_text(encoding="utf-8"))
                if [file_fingerprint(p) for p in images] != hashes or identity() != execution_identity:
                    raise ProviderError("试评期间输入或引擎已变化，结果已拒绝")
                usage = next((e.get("usage", {}) for e in reversed(event_list) if e.get("type") == "turn.completed"), {})
                return raw, {"phase": phase, "engine": execution_identity, "input_hashes": hashes,
                             "elapsed_seconds": round(time.monotonic() - started, 3), **usage}
            finally:
                _terminate(process)
