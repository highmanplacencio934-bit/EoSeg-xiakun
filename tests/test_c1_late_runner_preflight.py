"""Exercise the C1-Late runner's real YAML preflight without starting training."""

import subprocess
import sys
import unittest
from pathlib import Path


class C1LateRunnerPreflightTests(unittest.TestCase):
    def test_frozen_learning_rates_pass_the_real_preflight(self):
        root = Path(__file__).resolve().parents[1]
        runner = root / "configs/glas/run_protocol_b_c1_late_abl_noes_1000.sh"
        config = root / "configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_late_abl_noes_1000.yaml"
        script = runner.read_text(encoding="utf-8")
        marker = 'python - "${CONFIG}" <<\'PY\'\n'
        preflight = script.split(marker, 1)[1].split("\nPY\n", 1)[0]

        result = subprocess.run(
            [sys.executable, "-", str(config)],
            input=preflight,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("C1-Late configuration check passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
