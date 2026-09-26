"""Run: python -m unittest -v test_labs.py"""
import asyncio
from pathlib import Path
import tempfile
import unittest
import numpy as np
from attention import attention
from async_workers import run_cases
from eval_gate import Score, compare
from feature_join import point_in_time
from graph import topological_sort
from numerical import conv2d, finite_difference, logistic_loss_gradient
from retrieval import Document, answer, metrics, retrieve, rrf
from safe_action import ActionStore, SimulatedCrash

class RetrievalTests(unittest.TestCase):
    def test_tenant_before_ranking(self):
        docs = [Document("private", "b", "refund refund"), Document("public", "a", "refund receipt")]
        self.assertEqual([d.id for d in retrieve("refund", docs, "a")], ["public"])
        self.assertNotIn("private", str(answer("refund", docs, "a")))

    def test_revision_and_no_evidence(self):
        docs = [Document("a", "t", "old", 1), Document("a", "t", "new", 2)]
        self.assertEqual(retrieve("old", docs, "t"), [])
        self.assertEqual(answer("unknown", docs, "t")["status"], "no_evidence")
        self.assertEqual(retrieve("new", docs, "t")[0].revision, 2)

    def test_rrf_duplicate_not_boosted(self):
        self.assertEqual(rrf([["a", "a", "b"], ["b"]]), rrf([["a", "b"], ["b"]]))
        self.assertEqual(rrf([["a", "b"], ["b", "c"]])[0], "b")

    def test_metrics(self):
        result = metrics(["x", "a", "b"], {"a": 1, "b": 1, "c": 1}, 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["rr"], .5)
        self.assertEqual(metrics(["a", "a"], {"a": 1}, 2)["precision"], .5)
        self.assertIsNone(metrics([], {}, 3)["recall"])

    def test_ndcg_ideal(self):
        self.assertEqual(metrics(["a", "b"], {"a": 3, "b": 1}, 2)["ndcg"], 1)
        self.assertLess(metrics(["b", "a"], {"a": 3, "b": 1}, 2)["ndcg"], 1)

class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.x = self.rng.normal(size=(2, 4, 8))
        self.ws = [self.rng.normal(size=(8, 8)) / 4 for _ in range(4)]

    def test_single_head_reference(self):
        out, weights = attention(self.x, *self.ws, causal=False)
        q, k, v = [self.x @ w for w in self.ws[:3]]
        logits = q @ k.swapaxes(-1, -2) / np.sqrt(8)
        expected_weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        expected_weights /= expected_weights.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(out, expected_weights @ v @ self.ws[3], atol=1e-12)

    def test_multihead_manual_reference(self):
        out, _ = attention(self.x, *self.ws, heads=2, causal=False)
        heads = []
        projected = [self.x @ w for w in self.ws[:3]]
        for h in range(2):
            q, k, v = [p[:, :, h * 4:(h + 1) * 4] for p in projected]
            scores = q @ k.swapaxes(-1, -2) / 2
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            heads.append(probs @ v)
        np.testing.assert_allclose(out, np.concatenate(heads, axis=-1) @ self.ws[3], atol=1e-12)

    def test_future_cannot_change_past(self):
        old, _ = attention(self.x, *self.ws, heads=2)
        changed = self.x.copy()
        changed[:, 2:] += 100
        new, _ = attention(changed, *self.ws, heads=2)
        np.testing.assert_allclose(old[:, :2], new[:, :2], atol=1e-12)

    def test_padding_and_all_masked(self):
        valid = np.array([[True, True, False, False], [False] * 4])
        out, weights = attention(self.x, *self.ws, heads=2, valid=valid)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(out[1], 0)
        np.testing.assert_array_equal(out[0, 2:], 0)
        np.testing.assert_array_equal(weights[..., 2:], 0)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            attention(self.x, *self.ws, heads=3)

class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_and_accounting(self):
        active = peak = 0
        async def operation(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.001)
                return value
            finally:
                active -= 1
        results = await run_cases(((str(i), i) for i in range(30)), operation, workers=3, capacity=2)
        self.assertEqual({r.case_id for r in results}, {str(i) for i in range(30)})
        self.assertEqual(peak, 3)
        self.assertEqual(active, 0)

    async def test_errors_and_timeout(self):
        async def operation(value):
            if value == "slow":
                await asyncio.Event().wait()
            if value == "bad":
                raise ValueError("malformed")
            return 1
        rows = await run_cases([(x, x) for x in ["slow", "bad", "good"]], operation, timeout=.02)
        self.assertEqual({r.case_id: r.status for r in rows}, {"slow": "timeout", "bad": "error", "good": "ok"})

    async def test_cancellation_cleans_workers(self):
        entered, finished = asyncio.Event(), asyncio.Event()
        async def operation(value):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()
        task = asyncio.create_task(run_cases([("a", 1)], operation))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())

    async def test_duplicate_ids_fail_job(self):
        async def operation(value):
            return value
        with self.assertRaises(ExceptionGroup):
            await run_cases([("a", 1), ("a", 2)], operation)

class ActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "actions.db"
        self.store = ActionStore(self.path)
    def tearDown(self):
        self.store.close()
        self.temp.cleanup()
    def test_before_commit_rollback(self):
        with self.assertRaises(SimulatedCrash):
            self.store.execute("x", {"amount": 5}, "before_commit")
        self.assertEqual(self.store.count(), 0)
        self.store.execute("x", {"amount": 5})
        self.assertEqual(self.store.count(), 1)
    def test_after_commit_restart_and_retry(self):
        with self.assertRaises(SimulatedCrash):
            self.store.execute("x", {"amount": 5}, "after_commit")
        self.store.close()
        self.store = ActionStore(self.path)
        self.assertEqual(self.store.execute("x", {"amount": 5})["status"], "committed")
        self.assertEqual(self.store.count(), 1)
    def test_conflict_rejected(self):
        self.store.execute("x", {"amount": 5})
        with self.assertRaises(ValueError):
            self.store.execute("x", {"amount": 6})
        self.assertEqual(self.store.count(), 1)

class FeatureTests(unittest.TestCase):
    def test_future_late_and_missing(self):
        rows = point_in_time([("p1", "a", 100), ("p2", "b", 100)],
                             [(1, "a", 90, 95, 4), (2, "a", 99, 101, 100), (3, "a", 110, 90, 200)])
        self.assertEqual(rows, [("p1", "a", 1, 4.), ("p2", "b", None, None)])
    def test_equal_boundary_and_tie(self):
        rows = point_in_time([("p", "a", 100)], [(1, "a", 100, 100, 1), (2, "a", 100, 100, 2)])
        self.assertEqual(rows[0][2:], (2, 2.))

class GraphTests(unittest.TestCase):
    def test_dependencies_disconnected_duplicates(self):
        result = topological_sort(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("a", "b")])
        self.assertEqual(set(result), {"a", "b", "c", "d"})
        self.assertLess(result.index("a"), result.index("b"))
        self.assertLess(result.index("b"), result.index("c"))
    def test_cycle_and_missing_node(self):
        for nodes, edges in [(["a", "b"], [("a", "b"), ("b", "a")]), (["a"], [("a", "a")]), (["a"], [("a", "b")])]:
            with self.assertRaises(ValueError):
                topological_sort(nodes, edges)
    def test_empty(self):
        self.assertEqual(topological_sort([], []), [])

class NumericalTests(unittest.TestCase):
    def test_gradient(self):
        x, y, w = np.array([[1., 2.], [1., -1.], [1., .5]]), np.array([1., 0., 1.]), np.array([.2, -.3])
        _, gradient = logistic_loss_gradient(x, y, w)
        ref = finite_difference(lambda p: logistic_loss_gradient(x, y, p)[0], w)
        np.testing.assert_allclose(gradient, ref, atol=1e-8)
    def test_extreme_logits(self):
        loss, gradient = logistic_loss_gradient([[1], [-1]], [0, 1], [1000])
        self.assertTrue(np.isfinite(loss) and np.isfinite(gradient).all())
        self.assertAlmostEqual(loss, 1000)
    def test_asymmetric_kernel_not_flipped(self):
        x = np.arange(9).reshape(1, 1, 3, 3)
        kernel = np.array([[[[1, 2], [3, 4]]]])
        np.testing.assert_allclose(conv2d(x, kernel), [[[[27, 37], [57, 67]]]])
    def test_convolution_stride_padding_channels(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(2, 3, 4, 5))
        k = rng.normal(size=(2, 3, 2, 3))
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (2, 3), axis=(2, 3))[:, :, ::2, ::2]
        expected = np.einsum("ncijhw,ochw->noij", windows, k)
        np.testing.assert_allclose(conv2d(x, k, stride=2, padding=1), expected, atol=1e-12)

class GateTests(unittest.TestCase):
    def setUp(self):
        self.ids = [str(i) for i in range(20)]
        self.old = [Score(i, .7) for i in self.ids]
    def test_improvement_passes(self):
        self.assertEqual(compare(self.ids, self.old, [Score(i, .8) for i in self.ids])["decision"], "pass")
    def test_regression_fails(self):
        self.assertEqual(compare(self.ids, self.old, [Score(i, .5) for i in self.ids])["decision"], "fail")
    def test_uncertainty_inconclusive(self):
        new = [Score(i, .9 if n % 2 else .3) for n, i in enumerate(self.ids)]
        self.assertEqual(compare(self.ids, self.old, new)["decision"], "inconclusive")
    def test_critical_failure_overrides_quality(self):
        new = [Score(i, .9, critical_failure=i == "0") for i in self.ids]
        self.assertEqual(compare(self.ids, self.old, new)["reason"], "critical safety failure")
    def test_missing_duplicate_nan_wrong_ids_and_errors(self):
        bad_sets = [self.old[:-1], self.old + [self.old[0]],
                    [Score("wrong", .8)] + self.old[1:],
                    [Score("0", float("nan"))] + self.old[1:],
                    [Score("0", None, "timeout")] + self.old[1:]]
        for rows in bad_sets:
            self.assertEqual(compare(self.ids, self.old, rows)["decision"], "incomplete")
    def test_empty_never_passes(self):
        self.assertEqual(compare([], [], [])["decision"], "incomplete")

if __name__ == "__main__":
    unittest.main()
