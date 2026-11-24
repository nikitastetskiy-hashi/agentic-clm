"""GitHub Copilot as a model of the model steps (lib/copilot.py): what Copilot CLI is given for a question
and as the repair agent, what a refusal does to the step's next model, and that no test installs it."""
import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers
import enrich
from lib import agent, copilot, gateway, sandbox

# Stands in for Copilot CLI: records its arguments, environment, directory and question, and answers as
# `copilot --silent` does: the answer alone on the standard output, the usage in the file it was told. The
# model "gone" is refused as the CLI refuses one it does not serve. A wording request gets one bullet per item.
FAKE_CLI = """#!{python}
import json, os, sys
from pathlib import Path
args, prompt = sys.argv[1:], sys.stdin.read()
Path(os.environ["FAKE_CLI_LOG"]).write_text(json.dumps({{"args": args, "prompt": prompt,
    "env": sorted(k for k in os.environ if k.endswith(("_KEY", "_TOKEN"))),
    "token_ok": os.environ.get("COPILOT_GITHUB_TOKEN") == "ghs_copilot0123456789abcdef",
    "empty_directory": not os.listdir("."), "home": os.environ["HOME"], "cwd": os.getcwd()}}))
if args[args.index("--model") + 1] == "gone":
    sys.stderr.write('\\nExecution failed: Model "gone" is not available.\\n')
    sys.exit(1)
Path(args[args.index("--usage-output-file") + 1]).write_text(json.dumps({{"totalNanoAiu": 260977500}}))
reply = "OK"
try:
    items = json.loads(prompt.rsplit("\\n\\n---\\n\\n", 1)[1])["items"]
    reply = json.dumps({{"items": [{{"id": i["id"], "bullets": ["Adds what its release notes describe, in one plain sentence."]}} for i in items]}})
except (IndexError, ValueError, KeyError, TypeError):
    pass
print(reply + "\\n")
"""
KEYS = {"COPILOT_GITHUB_TOKEN": "ghs_copilot0123456789abcdef", "ICA_API_KEY": "sk-ica-0123456789abcdef",
        "GIT_TOKEN": "ghs_0123456789abcdef"}


class CopilotCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        fake = self.tmp / "copilot"
        fake.write_text(FAKE_CLI.format(python=sys.executable), encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.log = self.tmp / "cli-call.json"
        self.enterContext(mock.patch.dict(os.environ, {"COPILOT_CLI": str(fake), "FAKE_CLI_LOG": str(self.log), **KEYS}))

    def call(self) -> dict:
        return json.loads(self.log.read_text(encoding="utf-8"))

    def test_a_question_gets_an_empty_directory_one_read_only_tool_and_only_the_token(self):
        self.assertEqual(gateway.complete("copilot/claude-haiku-5.5", [{"role": "system", "content": "Be brief."},
                                                                        {"role": "user", "content": "Reply OK"}]), (200, "OK"))
        call = self.call()
        args = call["args"]
        self.assertEqual(args[args.index("--model") + 1], "claude-haiku-5.5")
        self.assertEqual(args[-2:], ["--available-tools", "view"])      # last: it takes every word after it
        for flag in ("--silent", "--no-ask-user", "--no-auto-update", "--no-custom-instructions", "--disable-builtin-mcps",
                     "--disallow-temp-dir"):
            self.assertIn(flag, args)
        self.assertNotIn("--allow-tool", args)                          # nothing to approve: it may not write
        self.assertEqual(args[args.index("--max-ai-credits") + 1], "30")
        self.assertEqual(call["prompt"], gateway.NO_TOOLS + "Be brief.\n\n---\n\nReply OK")
        self.assertEqual(call["env"], ["COPILOT_GITHUB_TOKEN"])         # not the gateway key, not the pipeline's token
        self.assertTrue(call["token_ok"] and call["empty_directory"])
        self.assertNotEqual(call["home"], os.environ.get("HOME"))
        self.assertAlmostEqual(copilot.ask("Reply OK", model="claude-haiku-5.5")[2], 0.261, places=3)   # AI credits

    def test_a_model_copilot_does_not_serve_hands_over_to_the_step_s_next_one(self):
        request = {"items": [{"id": "vault/2.1.0", "kind": "release", "max_bullets": 1, "max_chars": 230,
                              "source": {"notes": {"Features": ["Adds what its release notes describe"]}}}]}
        env = dict(os.environ, ENRICH_MODEL="copilot/gone", ENRICH_FALLBACK_MODEL="copilot/gpt-6-luna")
        answers, model, failures = enrich.ask_direct(request, env)
        self.assertEqual(model, "copilot/gpt-6-luna")
        self.assertEqual(failures, ['copilot/gone: HTTP 400: Copilot CLI: Execution failed: Model "gone" is not available'])
        self.assertEqual(list(answers), ["vault/2.1.0"])

    def test_without_the_token_copilot_is_not_run(self):
        with mock.patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": ""}):
            self.assertIsNone(gateway.endpoint("copilot/claude-haiku-5.5"))
            self.assertEqual(gateway.complete("copilot/claude-haiku-5.5", []), (None, "no COPILOT_GITHUB_TOKEN set"))
        self.assertFalse(self.log.exists())

    def test_the_preflight_asks_copilot_for_the_models_named(self):
        env = dict(os.environ, ENRICH_MODEL="copilot/claude-haiku-5.5", REVIEW_MODELS="copilot/gone", GITHUB_ACTIONS="")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(gateway.preflight(env), 1)
        self.assertRegex(out.getvalue(), r"copilot/claude-haiku-5\.5\s+accepted")
        self.assertRegex(out.getvalue(), r"copilot/gone\s+REFUSED by the gateway \(HTTP 400: Copilot CLI: .*not available\)")

    def test_the_repair_agent_on_copilot_reads_and_edits_but_runs_no_command(self):
        # The token is the job's own and cannot be kept from a command the CLI runs, so it runs none.
        with mock.patch.dict(os.environ, {"FIX_MODEL": "copilot/claude-sonnet-5.5", "FIX_FALLBACK_MODEL": "ica-gateway/gpt-6.1-sol"}):
            self.assertEqual(list(json.loads(agent.opencode_config())["provider"]), ["ica-gateway"])   # not opencode's to call
        ran = agent.run_first("repo-fixer", "fix it", models=["copilot/claude-sonnet-5.5"], answered=lambda code: code == 0,
                              cwd=self.tmp, log_file=self.tmp / "agent.log")
        self.assertEqual(ran, (0, "copilot/claude-sonnet-5.5"))
        call = self.call()
        args = call["args"]
        self.assertEqual(args[args.index("--available-tools") + 1:], ["view", "grep", "glob", "edit", "create"])
        self.assertEqual(args[args.index("--allow-tool") + 1], "write")
        self.assertFalse({"--allow-all-tools", "--allow-all", "--yolo", "--allow-all-paths", "--add-dir"} & set(args))
        self.assertEqual((args[args.index("--model") + 1], args[args.index("--max-ai-credits") + 1]), ("claude-sonnet-5.5", "500"))
        self.assertTrue(call["prompt"].endswith("\n\n---\n\n" + agent.NO_COMMANDS + "fix it"))
        self.assertEqual(call["env"], ["COPILOT_GITHUB_TOKEN"])
        self.assertEqual(Path(call["cwd"]).resolve(), self.tmp.resolve())
        self.assertIn("AI credits: 0.26", (self.tmp / "agent.log").read_text(encoding="utf-8"))

    def test_a_tool_that_runs_commands_or_fetches_is_never_given_whoever_asks(self):
        copilot.run("x", model="m", workspace=self.tmp, tools=("view", "bash", "web_fetch", "edit"), max_credits=30)
        args = self.call()["args"]
        self.assertEqual(args[args.index("--available-tools") + 1:], ["view", "edit"])

    def test_in_the_sandbox_the_agent_reads_the_token_from_its_input_alone(self):
        ran = []

        def without_sudo(cmd, *, cwd, env):   # what sandbox.command runs after sudo switches user
            line = [*(f"{k}={v}" for k, v in sorted(env.items()))]
            ran.append(line + cmd)
            return ["env", "-i", *line, "/bin/bash", "-c", 'cd "$1" && shift && exec "$@"', "sandbox", str(cwd), *cmd]

        work = self.tmp / "repo"
        (work / "output").mkdir(parents=True)
        (work / "leads-out").symlink_to("/etc")
        with mock.patch.dict(os.environ, {"SANDBOX_USER": "sandbox-user"}), \
                mock.patch.object(sandbox, "command", without_sudo), \
                mock.patch.object(sandbox, "give"), mock.patch.object(sandbox, "take"), \
                mock.patch.object(sandbox, "clear") as clear, \
                mock.patch.object(sandbox.pwd, "getpwnam", return_value=mock.Mock(pw_dir=str(self.tmp))):
            agent.run_first("repo-fixer", "fix it", models=["copilot/claude-sonnet-5.5"], answered=lambda code: True, cwd=work)
        call = self.call()
        self.assertTrue(call["token_ok"])                                     # the CLI has the token...
        self.assertFalse(any(v in arg for line in ran for arg in line for v in KEYS.values()))   # ...never on a command line
        self.assertNotIn(KEYS["COPILOT_GITHUB_TOKEN"], call["prompt"])
        self.assertTrue(call["prompt"].endswith("fix it"))
        self.assertEqual(call["env"], ["COPILOT_GITHUB_TOKEN"])
        self.assertEqual(Path(call["cwd"]).resolve(), work.resolve())
        self.assertFalse((work / "leads-out").is_symlink())                   # no link leads it out of its copy
        self.assertTrue(clear.called)


class Install(unittest.TestCase):
    def test_the_tests_never_install_it(self):
        with mock.patch.dict(os.environ, {"COPILOT_CLI": "", "COPILOT_CLI_HOME": "/nonexistent"}):
            with self.assertRaisesRegex(copilot.CopilotError, "COPILOT_CLI_INSTALL=0"):
                copilot.install()

    def test_npm_installs_the_pinned_version_and_sees_no_secret(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"COPILOT_CLI": "", "COPILOT_CLI_HOME": tmp, "COPILOT_CLI_INSTALL": "1", **KEYS}), \
                mock.patch.object(copilot.bob, "node_version", return_value=(22, 3)), \
                mock.patch.object(copilot.shutil, "which", return_value="/usr/bin/npm"), \
                mock.patch.object(copilot.subprocess, "run", return_value=mock.Mock(returncode=0, stderr="")) as npm:
            with self.assertRaisesRegex(copilot.CopilotError, "no copilot command"):   # the stand-in installed nothing
                copilot.install()
        cmd = npm.call_args.args[0]
        self.assertEqual(cmd[-1], f"@github/copilot@{copilot.VERSION}")
        self.assertIn("--registry=https://registry.npmjs.org/", cmd)
        self.assertEqual([k for k in npm.call_args.kwargs["env"] if k.endswith(("_KEY", "_TOKEN"))], [])


if __name__ == "__main__":
    unittest.main()
