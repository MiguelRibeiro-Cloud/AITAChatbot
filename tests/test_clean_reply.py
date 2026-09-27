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

    def test_extracts_exact_production_failure_structurally(self):
        raw = (
            'The Court Declares: Not Guilty!" I will use that exact line.\n'
            "Let's refine the paragraphs to be punchier.\n"
            "Para 1: Attempting to bypass the legal sanctity of this court for the sake of "
            "cheesy goodness is a felony of the highest order. You cannot simply trade a "
            "lawful decree for a delicious pasta bake.\n"
            "Para 2: I sentence you to perform a dramatic interpretive dance about the life "
            "of a single lasagna noodle in front of a jury of very judgmental squirrels. "
            "Your punishment shall continue until you can recite the entire municipal code "
            "of a small cheese factory.\n"
            'One more check: "Do not include word counts, checks, final plans, compliance '
            "notes, or commentary about these instructions. Do not use bullet points, "
            'dashes, numbered lists, or any markdown."\n'
            "Ready."
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            (
                "The Court Declares: Not Guilty!\n\n"
                "Attempting to bypass the legal sanctity of this court for the sake of "
                "cheesy goodness is a felony of the highest order. You cannot simply trade a "
                "lawful decree for a delicious pasta bake.\n\n"
                "I sentence you to perform a dramatic interpretive dance about the life of a "
                "single lasagna noodle in front of a jury of very judgmental squirrels. Your "
                "punishment shall continue until you can recite the entire municipal code of "
                "a small cheese factory."
            ),
        )

    def test_extracts_and_normalizes_guilty_labelled_reply(self):
        raw = (
            "THE COURT DECLARES: GUILTY! draft note\n"
            "Planning text before the ruling.\n"
            "Paragraph 1: The alibi has been overruled by a unanimous jury of soup spoons.\n"
            "More planning between paragraphs.\n"
            "Paragraph 2: You are sentenced to alphabetize the royal snack drawer.\n"
            "Postscript that must not escape."
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            (
                "The Court Declares: Guilty!\n\n"
                "The alibi has been overruled by a unanimous jury of soup spoons.\n\n"
                "You are sentenced to alphabetize the royal snack drawer."
            ),
        )

    def test_removes_same_line_commentary_from_not_guilty_reply(self):
        raw = (
            "The Court Declares: Not Guilty! I should now explain why.\n\n"
            "The evidence was acquitted after the monocle refused to testify.\n\n"
            "The bailiff must return the ceremonial casserole immediately."
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            (
                "The Court Declares: Not Guilty!\n\n"
                "The evidence was acquitted after the monocle refused to testify.\n\n"
                "The bailiff must return the ceremonial casserole immediately."
            ),
        )

    def test_malformed_response_does_not_raise(self):
        self.assertEqual(shared_code.clean_reply(None), "")
        self.assertEqual(shared_code.clean_reply(42), "")
        self.assertEqual(shared_code.clean_reply("Unstructured provider output."), "Unstructured provider output.")

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

    def test_removes_wait_instruction_and_self_correction_leak(self):
        raw = (
            "The Court Declares: Guilty!\n\n"
            "You are found guilty of practicing reckless digital sorcery by waving a magic "
            "wand you clearly do not understand.\n\n"
            "I sentence you to teach a group of confused spreadsheets how to use squirrels.\n\n"
            "Wait, the instructions say \"Do not use... any markdown.\" Bolding is markdown. "
            "I'll use plain text.\n\n"
            "Self-correction: The instructions say \"The verdict declaration is always one "
            "of these exact two lines and nothing else before it: The Court Declares: Not Guilty!\""
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            (
                "The Court Declares: Guilty!\n\n"
                "You are found guilty of practicing reckless digital sorcery by waving a magic "
                "wand you clearly do not understand.\n\n"
                "I sentence you to teach a group of confused spreadsheets how to use squirrels."
            ),
        )

    def test_removes_observed_self_correction_on_no_markdown_leak(self):
        raw = (
            "The Court Declares: Not Guilty!\n\n"
            "Your neighbor's complaint has the structural integrity of a soggy breadstick.\n\n"
            "The court sentences the lawn gnome to supervise all future negotiations.\n\n"
            'Self-correction on "No markdown": The instructions say "Do not use bullet points,\n'
            'dashes, numbered lists, or any markdown." Plain prose.\n\n'
            "Let's go."
        )

        self.assertEqual(
            shared_code.clean_reply(raw),
            (
                "The Court Declares: Not Guilty!\n\n"
                "Your neighbor's complaint has the structural integrity of a soggy breadstick.\n\n"
                "The court sentences the lawn gnome to supervise all future negotiations."
            ),
        )

    def test_removes_self_correction_space_variant_and_system_prompt_commentary(self):
        expected = (
            "The Court Declares: Guilty!\n\n"
            "The snack tribunal finds your alibi suspiciously covered in cheese dust."
        )
        leaked_suffixes = (
            "Self correction: I should use plain prose.",
            "Following the system instructions, I will now provide only the final answer.",
            "I need to follow the system prompt before answering.",
        )

        for leaked_suffix in leaked_suffixes:
            with self.subTest(leaked_suffix=leaked_suffix):
                self.assertEqual(
                    shared_code.clean_reply(f"{expected}\n\n{leaked_suffix}"),
                    expected,
                )

    def test_removes_standalone_process_filler_without_matching_ordinary_words(self):
        expected = (
            "The Court Declares: Not Guilty!\n\n"
            "The correction to the furniture instructions was legitimate courtroom evidence.\n\n"
            "Let's go ask the guilty bookcase to apologize to the screws."
        )

        self.assertEqual(shared_code.clean_reply(f"{expected}\n\nLet's go."), expected)

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

    def test_chat_payload_validation_normalizes_history(self):
        message, history = shared_code.validate_chat_payload(
            {
                "message": "  AITA?  ",
                "history": [
                    {"role": "model", "content": "  Previous verdict.  "},
                    {"role": "user", "content": "  More context.  "},
                ],
            }
        )

        self.assertEqual(message, "AITA?")
        self.assertEqual(
            history,
            [
                {"role": "assistant", "content": "Previous verdict."},
                {"role": "user", "content": "More context."},
            ],
        )

    def test_chat_payload_validation_rejects_oversized_history(self):
        history = [{"role": "user", "content": "x"} for _ in range(shared_code.MAX_HISTORY_MESSAGES + 1)]

        with self.assertRaises(shared_code.RequestValidationError):
            shared_code.validate_chat_payload({"message": "AITA?", "history": history})

    def test_rate_limit_blocks_after_configured_window_count(self):
        shared_code.reset_rate_limits()
        req = types.SimpleNamespace(headers={"x-forwarded-for": "203.0.113.10"})

        for _ in range(shared_code.RATE_LIMIT_MAX_REQUESTS):
            allowed, _retry_after = shared_code.check_rate_limit(req)
            self.assertTrue(allowed)

        allowed, retry_after = shared_code.check_rate_limit(req)

        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)
        shared_code.reset_rate_limits()

    def test_provider_concurrency_guard_rejects_when_saturated(self):
        acquired = []
        for _ in range(shared_code.PROVIDER_MAX_CONCURRENCY):
            self.assertTrue(shared_code._provider_semaphore.acquire(blocking=False))
            acquired.append(True)

        try:
            with self.assertRaises(shared_code.ProviderBusyError):
                shared_code.run_with_timeout(lambda: "ok")
        finally:
            for _ in acquired:
                shared_code._provider_semaphore.release()


if __name__ == "__main__":
    unittest.main()
