"""Проверка алгоритмов независимыми расчётами и эталонами задания."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from scripts.bigram_lm import BOS, EOS, UNK, BigramLM
from scripts.combine_candidate_scores import rank_candidates
from scripts.run_lab_03 import run_lab
from scripts.viterbi_alignment import viterbi_align


class ViterbiTests(unittest.TestCase):
    def test_fixture_and_alternative(self):
        data = json.loads((LAB_ROOT / "data/toy_alignment_costs.json").read_text(encoding="utf-8"))
        path, cost = viterbi_align(data["costs"])
        self.assertEqual(path, [0, 1, 1, 2, 3, 4])
        self.assertAlmostEqual(cost, 3.0)
        alternative = sum(data["costs"][t][s] for t, s in enumerate(data["known_alternative"]["path"]))
        self.assertAlmostEqual(alternative, 3.7)
        self.assertLess(cost, alternative)

    def test_tie_prefers_staying(self):
        # На последнем кадре равны стоимости предшественников 0 и 1.
        self.assertEqual(viterbi_align([[0, 0], [0, 0], [0, 0]]), ([0, 1, 1], 0.0))

    def test_custom_boundaries_and_one_frame(self):
        self.assertEqual(viterbi_align([[0, 2, 9], [0, 4, 3]], start_state=1, final_state=2), ([1, 2], 5.0))
        self.assertEqual(viterbi_align([[9, 2, 9]], start_state=1, final_state=1), ([1], 2.0))

    def test_matches_exhaustive_search(self):
        rng = random.Random(19)
        for _ in range(40):
            states = rng.randint(1, 4)
            frames = rng.randint(states, 6)
            costs = [[rng.randint(0, 9) for _ in range(states)] for _ in range(frames)]
            valid_paths = [p for p in itertools.product(range(states), repeat=frames)
                if p[0] == 0 and p[-1] == states - 1
                and all(b - a in (0, 1) for a, b in zip(p, p[1:]))]
            best = min(sum(costs[t][s] for t, s in enumerate(p)) for p in valid_paths)
            path, actual = viterbi_align(costs)
            with self.subTest(states=states, frames=frames):
                self.assertIn(tuple(path), valid_paths)
                self.assertEqual(actual, best)
                self.assertEqual(actual, sum(costs[t][s] for t, s in enumerate(path)))

    def test_invalid_or_unreachable_inputs(self):
        for costs, kwargs in [([], {}), ([[]], {}), ([[1], [1, 2]], {}),
                ([[float("nan")]], {}), ([[float("inf")]], {}),
                ([[0, 1]], {}), ([[0, 1], [0, 1]], {"start_state": 1, "final_state": 0}),
                ([[1]], {"start_state": -1})]:
            with self.subTest(costs=costs, kwargs=kwargs), self.assertRaises(ValueError):
                viterbi_align(costs, **kwargs)


class BigramTests(unittest.TestCase):
    def setUp(self):
        corpus = (LAB_ROOT / "data/toy_lm_corpus.txt").read_text(encoding="utf-8").splitlines()
        self.lm = BigramLM.train(corpus, k=0.5)

    def test_hand_counted_probabilities(self):
        self.assertEqual(self.lm.vocabulary, {"we", "they", "drink", "like", "green", "tea", "strong", "coffee", UNK, EOS})
        self.assertNotIn(BOS, self.lm.vocabulary)
        self.assertEqual(self.lm.context_counts["drink"], 4)
        self.assertEqual(self.lm.bigram_counts[("drink", "green")], 3)
        self.assertEqual(self.lm.bigram_counts[("drink", "strong")], 1)
        self.assertAlmostEqual(self.lm.bigram_probability("drink", "green"), 3.5 / 9)
        self.assertAlmostEqual(self.lm.bigram_probability("drink", "strong"), 1.5 / 9)
        self.assertAlmostEqual(self.lm.bigram_probability("drink", "tea"), 0.5 / 9)

    def test_distributions_sum_to_one(self):
        for context in self.lm.vocabulary | {BOS, "unseen_context"}:
            with self.subTest(context=context):
                probabilities = [self.lm.bigram_probability(context, w) for w in self.lm.vocabulary]
                self.assertTrue(all(0 < p <= 1 for p in probabilities))
                self.assertAlmostEqual(sum(probabilities), 1)

    def test_sentence_includes_eos_and_log10(self):
        tea = (4.5 / 12) * (2.5 / 9) * (3.5 / 9) * (5.5 / 10) * (5.5 / 10)
        coffee = (4.5 / 12) * (2.5 / 9) * (1.5 / 9) * (2.5 / 7) * (2.5 / 7)
        self.assertAlmostEqual(self.lm.sentence_logprob("we drink green tea"), math.log10(tea))
        self.assertAlmostEqual(self.lm.sentence_logprob("we drink strong coffee"), math.log10(coffee))

    def test_unknown_words_and_empty_sentence(self):
        self.assertEqual(self.lm.bigram_probability("drink", "cocoa"), self.lm.bigram_probability("drink", UNK))
        self.assertEqual(self.lm.bigram_probability("cocoa", EOS), self.lm.bigram_probability(UNK, EOS))
        # <s> -> <unk> -> </s>: невиданный контекст имеет равномерное распределение.
        self.assertAlmostEqual(self.lm.sentence_logprob("cocoa"), math.log10((0.5 / 12) * 0.1))
        self.assertAlmostEqual(self.lm.sentence_logprob(""), math.log10(0.5 / 12))


class RankingTests(unittest.TestCase):
    def test_weight_changes_winner_without_mutating_inputs(self):
        candidates = [{"id": "a", "acoustic_log_score": -1, "lm_log_score": -4},
                      {"id": "b", "acoustic_log_score": -2, "lm_log_score": -1}]
        self.assertEqual(rank_candidates(candidates, weight=0)[0]["id"], "a")
        ranked = rank_candidates(candidates, weight=0.5)
        self.assertEqual([r["id"] for r in ranked], ["b", "a"])
        self.assertEqual([r["combined_log_score"] for r in ranked], [-2.5, -3])
        self.assertTrue(all("combined_log_score" not in c for c in candidates))

    def test_invalid_scores_and_weights(self):
        for weight in (-1, float("inf"), float("nan")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                rank_candidates([], weight=weight)
        for candidate in ({}, {"acoustic_log_score": 0, "lm_log_score": float("nan")}):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                rank_candidates([candidate])


class IntegrationTests(unittest.TestCase):
    def test_all_four_artifacts_match_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_lab(LAB_ROOT / "data", output)
            for name in ("alignment_path.csv", "candidate_scores.csv", "summary.json", "viterbi_trellis.svg"):
                actual = (output / name).read_text(encoding="utf-8")
                expected = (LAB_ROOT / "outputs_reference" / name).read_text(encoding="utf-8")
                with self.subTest(artifact=name):
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
