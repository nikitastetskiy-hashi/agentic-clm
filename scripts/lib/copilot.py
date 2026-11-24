"""GitHub Copilot through Copilot CLI, GitHub's command-line client: a model for the steps that ask a
question (the wording, the pull request review), named `copilot/<model id>` and paid from the Copilot
allowance of the repository's owner.

On GitHub Actions the CLI signs in with the workflow's own token, in a job that has the
`copilot-requests: write` permission, so there is no key to store: the workflows pass that token as
COPILOT_GITHUB_TOKEN, and only a step whose model is named `copilot/...` uses it.

Copilot CLI is an agent, and a question needs none of that. Each one runs in an empty directory with a
home of its own, the question on the standard input and the answer alone on the standard output, with
one read-only tool (the CLI sends the description of every tool it has with the question, five times
what the question itself costs, and an empty tool list leaves it all of them), no MCP server, no custom
instructions, no update of itself, and a limit on the AI credits it may spend. The process sees the
token and no other variable named like a credential.

Installed on demand, as Bob Shell is (lib/bob.py): the pinned VERSION from the npm registry, where a
published version cannot be replaced (Node 22 or later). COPILOT_CLI_INSTALL=0 turns that off (the
tests do).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lib import bob, redact

VERSION = "1.0.115"
PACKAGE = "@github/copilot"
TOKEN_VAR = "COPILOT_GITHUB_TOKEN"   # the first variable Copilot CLI looks for a token in
MAX_CREDITS = 30                     # per question: the smallest limit the CLI takes (an AI credit is one US
                                     # cent). An answer being written is finished, so it cuts no answer short
REFUSED = 400                        # the CLI said no: another model may still be worth asking


class CopilotError(RuntimeError):
    pass


def install_dir() -> Path:
    return Path(os.environ.get("COPILOT_CLI_HOME") or Path.home() / ".cache" / "auto-pptx" / f"copilot-cli-{VERSION}")


def executable() -> Path | None:
    """The installed Copilot CLI, if any (COPILOT_CLI names another one, e.g. for the tests)."""
    if os.environ.get("COPILOT_CLI"):
        return Path(os.environ["COPILOT_CLI"])
    path = install_dir() / "node_modules" / ".bin" / "copilot"
    return path if path.exists() else None


def without_secrets(env) -> dict:
    return {k: v for k, v in env.items() if k not in redact.SECRET_VARS and not redact.SECRET_NAME.search(k)}


def install() -> Path:
    """Install the pinned Copilot CLI into install_dir(); returns its executable. Raises CopilotError."""
    found = executable()
    if found:
        return found
    if os.environ.get("COPILOT_CLI_INSTALL") == "0":
        raise CopilotError("Copilot CLI is not installed, and installing it is turned off (COPILOT_CLI_INSTALL=0)")
    node = bob.node_version()
    if not node or node < (22, 0) or not shutil.which("npm"):
        raise CopilotError(f"Copilot CLI needs Node 22 or later and npm; this machine has "
                           f"{'Node ' + '.'.join(map(str, node)) if node else 'no Node'}")
    target = install_dir()
    target.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["npm", "install", "--prefix", str(target), "--registry=https://registry.npmjs.org/",
                           "--no-audit", "--no-fund", "--loglevel=error", f"{PACKAGE}@{VERSION}"],
                          capture_output=True, text=True, timeout=900, env=without_secrets(os.environ))
    if done.returncode != 0:
        raise CopilotError(f"npm could not install {PACKAGE}@{VERSION}: {done.stderr.strip()[-300:]}")
    found = executable()
    if not found:
        raise CopilotError(f"{PACKAGE}@{VERSION} installed, but no copilot command came with it")
    return found


def ask(prompt: str, *, model: str, timeout: int = 600, env=None) -> tuple[int | None, str, float]:
    """One question to `model` through Copilot CLI. Returns (200, its answer, the AI credits it cost),
    (REFUSED, why, 0.0) when the CLI said no (a model it does not serve, a seat or a policy that refuses),
    or (None, why, 0.0) when nothing answered."""
    env = os.environ if env is None else env
    token = env.get(TOKEN_VAR, "")
    if not token:
        return None, f"no {TOKEN_VAR} set", 0.0
    try:
        cli = install()
    except CopilotError as exc:
        return None, str(exc), 0.0
    with tempfile.TemporaryDirectory(prefix="copilot-ask-") as tmp:
        home, work, usage = Path(tmp) / "home", Path(tmp) / "work", Path(tmp) / "usage.json"
        home.mkdir()
        work.mkdir()
        # --available-tools takes every word after it, so it comes last.
        cmd = [str(cli), "--silent", "--no-ask-user", "--no-auto-update", "--no-custom-instructions",
               "--disable-builtin-mcps", "--model", model, "--max-ai-credits", str(MAX_CREDITS),
               "--usage-output-file", str(usage), "--available-tools", "view"]
        try:
            done = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=str(work),
                                  env=without_secrets(env) | {"HOME": str(home), TOKEN_VAR: token})
        except subprocess.TimeoutExpired:
            return None, f"Copilot CLI did not finish in {timeout}s", 0.0
        try:   # counted in billionths of a credit
            credits = json.loads(usage.read_text(encoding="utf-8"))["totalNanoAiu"] / 1e9
        except (OSError, ValueError, KeyError, TypeError):
            credits = 0.0
    answer = done.stdout.strip()
    if done.returncode == 0 and answer:
        print(f"copilot: {model} answered for {credits:.2f} AI credit(s)", file=sys.stderr)
        return 200, answer, credits
    said = next((line.strip() for line in done.stderr.splitlines() if line.strip()), "") or f"exit {done.returncode}"
    return REFUSED, " ".join(redact.scrub(f"Copilot CLI: {said}", env).split()).rstrip(" .")[:300], 0.0
