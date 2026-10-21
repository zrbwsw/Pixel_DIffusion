"""Check SSH transport without a network connection, GPU, or training process."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_on_gpu.sh"


class GPUTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-transport-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project with spaces (test)"
        self.script = self.project / "scripts/run_on_gpu.sh"
        self.script.parent.mkdir(parents=True)
        self.script.write_text(SOURCE.read_text())
        self.capture = self.root / "ssh_connection.json"
        self.config_queries = self.root / "ssh_config_queries.txt"
        self.injected = self.root / "unexpected_shell_execution"

        # The fake SSH executes the quoted remote command locally, exactly as a
        # remote shell would. It records connection argv separately from ssh -G.
        ssh = self.root / "ssh"
        ssh.write_text(f"#!{sys.executable}\n" + '''
import json, os, subprocess, sys
args = sys.argv[1:]
if args[0] == "-G":
    alias = args[1]
    with open(os.environ["MOCK_CONFIG_QUERIES"], "a") as stream:
        stream.write(alias + "\\n")
    configured = alias == os.environ.get("MOCK_ALIAS")
    print("hostname " + ("h.pjlab.org.cn" if configured else alias))
    raise SystemExit(0)
with open(os.environ["MOCK_CAPTURE"], "w") as stream:
    json.dump(args, stream)
raise SystemExit(subprocess.call(["bash", "-c", args[-1]]))
''')
        hostname = self.root / "hostname"
        hostname.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "${MOCK_HOSTNAME:-preparation-node}"\n'
        )
        ssh.chmod(0o755)
        hostname.chmod(0o755)
        self.env = os.environ.copy()
        self.env.pop("GPU_SSH_KEY", None)
        self.env.update({
            "PATH": str(self.root) + os.pathsep + self.env["PATH"],
            "MOCK_CAPTURE": str(self.capture),
            "MOCK_CONFIG_QUERIES": str(self.config_queries),
            "MOCK_ALIAS": "",
            "MOCK_HOSTNAME": "preparation-node",
        })
        self.values = [
            "space value", "BaseT-(1-t,t)", "quotes'\"", "$HOME", "",
            "line\nnext", f"$(touch {self.injected})", f"`touch {self.injected}`",
        ]
        self.command = [
            sys.executable, "-c",
            "import os,sys,json; print(json.dumps([os.getcwd(),sys.argv[1:]]))",
            *self.values,
        ]

    def run_script(self, *args, env=None):
        result = subprocess.run(
            ["bash", str(self.script), *args],
            env={**self.env, **(env or {})},
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.injected.exists(), "Argument was executed as shell code")
        return result

    def assert_command_output(self, result):
        self.assertEqual(json.loads(result.stdout), [str(self.project), self.values])

    def test_remote_arguments_and_project_directory_are_preserved(self):
        result = self.run_script("agent-gpu", *self.command)
        self.assert_command_output(result)
        argv = json.loads(self.capture.read_text())
        self.assertEqual(argv[0], "-T")
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertEqual(
            argv[-2],
            "agent-gpu-10656330-3cc1d-cfcd2.gushishen+root.ailab-datafactory.pod@h.pjlab.org.cn",
        )
        self.assertEqual(argv[argv.index("-p") + 1], "22")
        self.assertEqual(argv[argv.index("-i") + 1], "/data/.ssh/id_ed25519_pjlab_20261013")

    def test_configured_gpu2_alias_takes_precedence(self):
        result = self.run_script("agent-gpu2", *self.command, env={"MOCK_ALIAS": "agent-gpu-2"})
        self.assert_command_output(result)
        argv = json.loads(self.capture.read_text())
        self.assertEqual(argv[-2], "agent-gpu-2")
        self.assertNotIn("-i", argv)  # Keep the configured SSH identity.
        self.assertNotIn("-p", argv)  # Keep the configured SSH port.

    def test_gpu2_fallback_and_identity_override(self):
        result = self.run_script(
            "agent-gpu-2", *self.command, env={"GPU_SSH_KEY": "/tmp/key with spaces"},
        )
        self.assert_command_output(result)
        argv = json.loads(self.capture.read_text())
        self.assertEqual(
            argv[-2],
            "agent-gpu-2-62133761-f83e2-cfcd2.gushishen+root.ailab-datafactory.pod@h.pjlab.org.cn",
        )
        self.assertEqual(argv[argv.index("-i") + 1], "/tmp/key with spaces")

    def test_dry_run_never_connects_or_executes_the_command(self):
        result = self.run_script("--dry-run", "agent-gpu2", *self.command)
        self.assertFalse(self.capture.exists())
        self.assertIn("ssh -T", result.stdout)
        self.assertIn("agent-gpu-2-62133761-", result.stdout)

    def test_matching_gpu_executes_locally_without_any_ssh(self):
        for host, hostname in (
            ("agent-gpu", "gpu-lg-cmc-h-h200-0029"),
            ("agent-gpu2", "gpu-lg-cmc-h-h200-0257"),
        ):
            with self.subTest(host=host):
                result = self.run_script(host, *self.command, env={"MOCK_HOSTNAME": hostname})
                self.assert_command_output(result)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.config_queries.exists())

    def test_interactive_dry_run_requests_tty_without_connecting(self):
        result = self.run_script("--dry-run", "agent-gpu")
        self.assertIn("ssh -t", result.stdout)
        self.assertIn("bash", result.stdout)
        self.assertFalse(self.capture.exists())

    def test_invalid_host_is_rejected_before_ssh(self):
        result = subprocess.run(
            ["bash", str(self.script), "unknown-host"],
            env=self.env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.config_queries.exists())


if __name__ == "__main__":
    unittest.main()
