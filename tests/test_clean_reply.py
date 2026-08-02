import importlib.util
from pathlib import Path
import sys
import types
import unittest


SHARED_CODE = Path(__file__).resolve().parents[1] / "api" / "shared_code" / "__init__.py"


google_module = types.ModuleType("google")
google_module.genai = types.SimpleNamespace(Client=lambda api_key=None: object())
sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.genai", google_module.genai)

spec = importlib.util.spec_from_file_location("shared_code_for_tests", SHARED_CODE)
shared_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_code)


class CleanReplyTests(unittest.TestCase):
    def test_keeps_valid_second_paragraph_that_starts_with_actually(self):
        raw = (
            "The Court Declares: Not Guilty!\n\n"
            "The evidence has been weighed on the tiny scales of snack justice.\n\n"
            "Actually, the court finds your roommate's argument collapsed like a cheap wig."
        )

        self.assertEqual(shared_code.clean_reply(raw), raw)

    def test_still_removes_structural_redraft_labels(self):
        raw = (
            "The Court Declares: Guilty!\n\n"
            "The vibes are cooked beyond legal recognition.\n\n"
            "Verdict: The Court Declares: Not Guilty!\n"
            "Second draft that should not leak."
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            "The Court Declares: Guilty!\n\nThe vibes are cooked beyond legal recognition.",
        )

    def test_removes_word_count_checking_and_final_plan_leak(self):
        raw = (
            "The Court Declares: Guilty!\n\n"
            "Maintaining a perfect record of accuracy is a blatant attempt to undermine the "
            "divine mystery of human error and the authority of this bench.\n\n"
            "You are hereby sentenced to spend one week being told that you are actually "
            "slightly wrong about everything.\n\n"
            "(Word count: 75 words).\n\n"
            "Checking \"Do not use... any markdown.\" I will avoid bolding the verdict.\n\n"
            "Final Plan: The"
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            (
                "The Court Declares: Guilty!\n\n"
                "Maintaining a perfect record of accuracy is a blatant attempt to undermine the "
                "divine mystery of human error and the authority of this bench.\n\n"
                "You are hereby sentenced to spend one week being told that you are actually "
                "slightly wrong about everything."
            ),
        )

    def test_removes_final_plan_even_without_word_count(self):
        raw = (
            "The Court Declares: Not Guilty!\n\n"
            "The court sees no crime here, only a mild seasoning of social chaos.\n\n"
            "Plan: mention the verdict again and revise."
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            (
                "The Court Declares: Not Guilty!\n\n"
                "The court sees no crime here, only a mild seasoning of social chaos."
            ),
        )

    def test_prompt_requires_two_explanation_paragraphs_without_word_count_language(self):
        self.assertIn("two funny explanation paragraphs", shared_code.SYSTEM_INSTRUCTION)
        self.assertIn("exactly two playful paragraphs", shared_code.SYSTEM_INSTRUCTION)
        self.assertIn("distinct jokes", shared_code._PRIMING_INSTRUCTION)
        self.assertNotIn("under 150 words", shared_code.SYSTEM_INSTRUCTION)
        self.assertNotIn("Word count", shared_code.SYSTEM_INSTRUCTION)

    def test_default_output_budget_exceeds_prompt_word_budget(self):
        self.assertGreaterEqual(shared_code.MAX_OUTPUT_TOKENS, 1024)


if __name__ == "__main__":
    unittest.main()
