"""GitHub Copilot through Copilot CLI, GitHub's command-line client: a model of every model step (the
wording, the pull request review, the repair agent), named `copilot/<model id>` and paid from the
Copilot allowance of the repository's owner.

On GitHub Actions the CLI signs in with the workflow's own token, in a job that has the
`copilot-requests: write` permission, so there is no key to store: the workflows pass that token as
COPILOT_GITHUB_TOKEN, and only a step whose model is named `copilot/...` uses it.

Copilot CLI is an agent. A question needs none of that: it runs in an empty directory with one
read-only tool (the CLI sends the description of every tool it has with the question, five times what
the question itself costs, and an empty tool list leaves it all of them). As the repair agent it reads
and edits the files of its directory and nothing else. It never gets the tool that runs commands: the
token cannot be kept from a command the CLI runs, and it is the job's own, so no command may run next
to it (the pipeline runs the tests itself, as it does after Bob, lib/bob.py). Every run has a home of
its own, the instructions on the standard input and the answer alone on the standard output, no MCP
server, no custom instructions, no update of itself, no file outside its directory (the CLI refuses
them, links included), and a limit on the AI credits it may spend. The process sees the token and no
other variable named like a credential. In the self-heal sandbox (lib/sandbox.py) the agent runs as
the sandbox user, alone, and reads the token from its standard input.

Installed on demand, as Bob Shell is: the pinned VERSION from the npm registry, where a published
version cannot be replaced (Node 22 or later). COPILOT_CLI_INSTALL=0 turns that off (the tests do).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lib import bob, redact, sandbox

VERSION = "1.0.115"
PACKAGE = "@github/copilot"
TOKEN_VAR = "COPILOT_GITHUB_TOKEN"   # the first variable Copilot CLI looks for a token in
READ_TOOLS = ("view", "grep", "glob")
EDIT_TOOLS = ("edit", "create")      # never "bash": no command may run next to the token
# An AI credit is one US cent. A question may spend the smallest limit the CLI takes; an answer being
# written is finished, so the limit cuts no answer short. A repair may spend what a Bob repair may.
QUESTION_CREDITS = 30
AGENT_CREDITS = 500
REFUSED = 400                        # the CLI said no: another model may still be worth asking
# The agent in the sandbox: the token arrives as the first line of the standard input, the instructions after it.
WITH_TOKEN_ON_STDIN = f'IFS= read -r {TOKEN_VAR} && export {TOKEN_VAR} && exec "$@"'


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
    """One question to `model`, in an empty directory with one read-only tool. Returns what run() does."""
    with tempfile.TemporaryDirectory(prefix="copilot-ask-") as empty:
        return run(prompt, model=model, workspace=Path(empty), tools=("view",), max_credits=QUESTION_CREDITS,
                   timeout=timeout, env=env)


def run(prompt: str, *, model: str, workspace: Path, tools: tuple[str, ...], max_credits: int, timeout: int = 600,
        env=None) -> tuple[int | None, str, float]:
    """Run Copilot CLI on `prompt` with `model` in `workspace`, with only `tools` (among READ_TOOLS and
    EDIT_TOOLS). Returns (200, its last message, the AI credits it cost), (REFUSED, why, 0.0) when the CLI
    said no (a model it does not serve, a seat or a policy that refuses), or (None, why, 0.0) when
    nothing answered."""
    env = os.environ if env is None else env
    token = env.get(TOKEN_VAR, "")
    if not token:
        return None, f"no {TOKEN_VAR} set", 0.0
    try:
        cli = install()
    except CopilotError as exc:
        return None, str(exc), 0.0
    tools = tuple(t for t in tools if t in READ_TOOLS + EDIT_TOOLS)
    edits = any(t in EDIT_TOOLS for t in tools)
    # In the sandbox an agent works on the sandbox's own copy of the repository, so it must run as that
    # user; a question needs no workspace of anyone's and runs as it is.
    boxed = sandbox.active(env) and edits
    home = Path(tempfile.mkdtemp(prefix="copilot-home-"))
    usage = home / "usage.json"
    # --allow-tool and --available-tools take every word after them, so they come last.
    cmd = [str(cli), "--silent", "--no-ask-user", "--no-auto-update", "--no-custom-instructions",
           "--disable-builtin-mcps", "--disallow-temp-dir", f"--secret-env-vars={TOKEN_VAR}", "--model", model,
           "--max-ai-credits", str(max_credits), "--usage-output-file", str(usage),
           *(("--allow-tool", "write") if edits else ()), "--available-tools", *tools]
    credits = 0.0
    try:
        if boxed:
            sandbox.give(home)
            sandbox.clear()              # no other process of the sandbox user while the CLI holds the token
            sandbox.unlink_links(workspace)
            done = sandbox.run(["/bin/bash", "-c", WITH_TOKEN_ON_STDIN, "copilot", *cmd], cwd=workspace,
                               env=sandbox.environment(env) | {"HOME": str(home)}, input=f"{token}\n{prompt}",
                               capture_output=True, text=True, timeout=timeout)
        else:
            done = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=str(workspace),
                                  env=without_secrets(env) | {"HOME": str(home), TOKEN_VAR: token})
    except subprocess.TimeoutExpired:
        return None, f"Copilot CLI did not finish in {timeout}s", 0.0
    finally:
        if boxed:
            sandbox.clear()
            sandbox.take(home)
        try:   # counted in billionths of a credit
            credits = json.loads(usage.read_text(encoding="utf-8"))["totalNanoAiu"] / 1e9
        except (OSError, ValueError, KeyError, TypeError):
            pass
        shutil.rmtree(home, ignore_errors=True)
    answer = done.stdout.strip()
    if done.returncode == 0 and answer:
        print(f"copilot: {model} answered for {credits:.2f} AI credit(s)", file=sys.stderr)
        return 200, answer, credits
    said = next((line.strip() for line in done.stderr.splitlines() if line.strip()), "") or f"exit {done.returncode}"
    return REFUSED, " ".join(redact.scrub(f"Copilot CLI: {said}", env).split()).rstrip(" .")[:300], 0.0
