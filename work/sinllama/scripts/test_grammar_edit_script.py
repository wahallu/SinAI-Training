#!/usr/bin/env python3
"""Dependency-free unit tests for the ByT5 v02 edit-script boundary."""

import json
import unittest

from grammar_edit_script import (
    apply_edit_script,
    make_edit_script,
    sanitize_generated_token_ids,
)


class GrammarEditScriptTests(unittest.TestCase):
    def round_trip(self, source: str, target: str) -> None:
        result = apply_edit_script(source, make_edit_script(source, target))
        self.assertEqual(result.status, "KEEP" if source == target else "APPLIED")
        self.assertEqual(result.text, target)

    def test_keep(self):
        self.round_trip("මෙය නිවැරදි වාක්‍යයකි.", "මෙය නිවැරදි වාක්‍යයකි.")

    def test_replace_insert_delete_and_multiple_edits(self):
        self.round_trip("කෙරුනි.", "කෙරුණි.")
        self.round_trip("ලක්ෂ10 ක්", "ලක්ෂ 10 ක්")
        self.round_trip("අද අද පැමිණියේය.", "අද පැමිණියේය.")
        self.round_trip("ඔහුන් සදහන් කලේය.", "ඔවුන් සඳහන් කළේය.")

    def test_invalid_old_text_is_rejected_to_keep(self):
        source = "කෙරුනි."
        script = json.dumps([{"s": 0, "e": 6, "o": "වැරදි", "n": "කෙරුණි"}], ensure_ascii=False)
        result = apply_edit_script(source, script)
        self.assertEqual((result.status, result.text), ("INVALID", source))

    def test_latin_name_deletion_is_rejected(self):
        source = "Mark Carney මහතා පැමිණියේය."
        result = apply_edit_script(source, make_edit_script(source, "මහතා පැමිණියේය."))
        self.assertEqual((result.status, result.text), ("REJECTED", source))
        self.assertIn("latin_span_mutation", result.reasons)

    def test_number_change_is_rejected_but_spacing_is_allowed(self):
        source = "මුදල රුපියල් 300කි."
        changed = apply_edit_script(source, make_edit_script(source, "මුදල රුපියල් 400කි."))
        self.assertEqual(changed.status, "REJECTED")
        self.round_trip("ලක්ෂ10 ක්", "ලක්ෂ 10 ක්")

    def test_unicode_format_control_change_is_rejected(self):
        source = "ක‍්ෂේත්‍රය"
        target = "ක‍්‍ෂේත්‍රය"
        result = apply_edit_script(source, make_edit_script(source, target))
        self.assertEqual((result.status, result.text), ("REJECTED", source))
        self.assertIn("unicode_format_control_mutation", result.reasons)

    def test_missing_eos_is_rejected(self):
        source, target = "කෙරුනි.", "කෙරුණි."
        result = apply_edit_script(source, make_edit_script(source, target), generation_finished=False)
        self.assertEqual((result.status, result.text), ("REJECTED", source))

    def test_trainer_negative_padding_is_sanitized_for_byt5(self):
        self.assertEqual(
            sanitize_generated_token_ids([[5, 6, 1, -100], [5, 1, -100, -100]], 0, 384),
            [[5, 6, 1, 0], [5, 1, 0, 0]],
        )

    def test_out_of_vocabulary_ids_are_also_sanitized(self):
        self.assertEqual(sanitize_generated_token_ids([[5, 9999999]], 0, 384), [[5, 0]])


if __name__ == "__main__":
    unittest.main()
