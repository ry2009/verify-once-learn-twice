from __future__ import annotations

import ast
import unittest

from ttRL.feedback import normalize_completion


def _parses_as_body(body: str) -> bool:
    try:
        ast.parse(f"def _f():\n{body}\n")
    except SyntaxError:
        return False
    return True


class NormalizeCompletionTests(unittest.TestCase):
    def test_multiline_def_body_not_truncated(self) -> None:
        text = (
            "def remove_duplicates(numbers):\n"
            "seen = set()\n"
            "result = []\n"
            "for num in numbers:\n"
            "    if num not in seen:\n"
            "        result.append(num)\n"
            "        seen.add(num)\n"
            "return result\n"
        )
        body = normalize_completion(text, entry_point="remove_duplicates")
        self.assertIn("result = []", body)
        self.assertIn("for num in numbers:", body)
        self.assertIn("return result", body)
        self.assertTrue(_parses_as_body(body))

    def test_inline_def_still_supported(self) -> None:
        body = normalize_completion("def f(x): return x + 1", entry_point="f")
        self.assertEqual(body.strip(), "return x + 1")
        self.assertTrue(_parses_as_body(body))

    def test_unindented_teacher_block_repaired(self) -> None:
        text = (
            "seen = set()\n"
            "result = []\n"
            "for num in numbers:\n"
            "if num not in seen:\n"
            "result.append(num)\n"
            "seen.add(num)\n"
            "return result\n"
        )
        body = normalize_completion(text, entry_point="remove_duplicates")
        self.assertIn("for num in numbers:", body)
        self.assertIn("if num not in seen:", body)
        self.assertTrue(_parses_as_body(body))

    def test_try_block_not_dropped(self) -> None:
        text = (
            "try:\n"
            "    s = s[-1] + s[:-1]\n"
            "except Exception:\n"
            "    pass\n"
            "return s[::3]\n"
        )
        body = normalize_completion(text, entry_point="stringop_000")
        self.assertIn("try:", body)
        self.assertIn("except Exception:", body)
        self.assertTrue(_parses_as_body(body))

    def test_repeated_real_ops_are_preserved(self) -> None:
        text = (
            "x = torch.nn.functional.mish(x)\n"
            "x = torch.nn.functional.mish(x)\n"
            "return x\n"
        )
        body = normalize_completion(text, entry_point="optimized_forward")
        self.assertEqual(body.count("torch.nn.functional.mish(x)"), 2)
        self.assertTrue(_parses_as_body(body))

    def test_trivial_return_loop_is_trimmed(self) -> None:
        text = "return x\nreturn x\nreturn x\n"
        body = normalize_completion(text, entry_point="f")
        self.assertEqual(body.strip(), "return x")
        self.assertTrue(_parses_as_body(body))


if __name__ == "__main__":
    unittest.main()
