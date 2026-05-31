from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import _JsonWithRepair, _repair_tool_call_arguments_in_place


class TestAgentJsonRepair(unittest.TestCase):
    def test_repairs_missing_colon_before_object_value(self):
        raw = '{"name":"A","value" {"status":"ok","shape":[2,3]}}'

        parsed = _JsonWithRepair.loads(raw)

        self.assertEqual(parsed["name"], "A")
        self.assertEqual(parsed["value"]["status"], "ok")
        self.assertEqual(parsed["value"]["shape"], [2, 3])

    def test_tool_call_arguments_are_normalized_after_repair(self):
        message = {
            "tool_calls": [
                {
                    "function": {
                        "name": "workspace_set",
                        "arguments": '{"name":"A","value" {"status":"ok"}}',
                    }
                }
            ]
        }

        changed = _repair_tool_call_arguments_in_place(message)
        args = message["tool_calls"][0]["function"]["arguments"]

        self.assertTrue(changed)
        self.assertEqual(json.loads(args), {"name": "A", "value": {"status": "ok"}})

    def test_repairs_workspace_snippet_code_with_unescaped_quotes_and_newlines(self):
        raw = '''{"code": "import numpy as np
RESULT = {"status": "ok", "n": int(A.shape[0])}", "array_refs": {"A": "A"}, "timeout_s": 10.0}'''

        parsed = _JsonWithRepair.loads(raw)

        self.assertEqual(parsed["array_refs"], {"A": "A"})
        self.assertEqual(parsed["timeout_s"], 10.0)
        self.assertIn('RESULT = {"status": "ok", "n": int(A.shape[0])}', parsed["code"])

    def test_tool_call_arguments_normalize_workspace_snippet_code(self):
        message = {
            "tool_calls": [
                {
                    "function": {
                        "name": "run_python_workspace_snippet",
                        "arguments": '''{"code": "b = np.ones(A.shape[0])
RESULT = {"saved_as": "b", "length": int(b.shape[0])}", "array_refs": {"A": "A"}}''',
                    }
                }
            ]
        }

        changed = _repair_tool_call_arguments_in_place(message)
        args = json.loads(message["tool_calls"][0]["function"]["arguments"])

        self.assertTrue(changed)
        self.assertEqual(args["array_refs"], {"A": "A"})
        self.assertIn('RESULT = {"saved_as": "b", "length": int(b.shape[0])}', args["code"])

    def test_repairs_workspace_snippet_code_when_code_is_last_argument(self):
        raw = '''{"array_refs": {"A": "A"}, "code": "RESULT = {"status": "ok"}"}'''

        parsed = _JsonWithRepair.loads(raw)

        self.assertEqual(parsed["array_refs"], {"A": "A"})
        self.assertEqual(parsed["code"], 'RESULT = {"status": "ok"}')


if __name__ == "__main__":
    unittest.main()
