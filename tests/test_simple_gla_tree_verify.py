#!/usr/bin/env python3
"""Unit tests for the Simple GLA tree-verify Triton kernel.

The new kernel ``fused_recurrent_simple_gla_update`` is the EAGLE3
verify path's analogue of upstream ``fused_recurrent_simple_gla``,
adding snapshot-and-replay support so each candidate token in the
draft tree attends only to its ancestors.

These tests build small synthetic tree topologies, run the kernel,
and compare against a pure-PyTorch reference implementation that
performs the per-branch recurrence in eager mode.

Run:
    python3 -m unittest tests.test_simple_gla_tree_verify

Requires CUDA: skipped automatically on CPU-only hosts (most of
this repo's local dev mirror).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))


@unittest.skipUnless(
    torch.cuda.is_available(), "Triton kernel requires CUDA; skipped on CPU."
)
class TestSimpleGLATreeVerify(unittest.TestCase):
    """Triton kernel correctness checks vs an eager Python reference.

    The Simple GLA recurrence is
        h_t = exp(g_gamma[h]) * h_{t-1} + k_t v_t^T
        o_t = scale * q_t^T h_t
    and under tree verify the parent of token t is given by
    ``retrieve_parent_token[t]``; step 0 reads from the initial state.
    """

    @staticmethod
    def _reference(q, k, v, g_gamma, scale, h0, retrieve_parent_token):
        """Pure-PyTorch reference. All math in float32 to match kernel.

        Shapes:
            q, k:   (T, H, K)
            v:      (T, HV, V) with HV == H for SALA's no-GQA case
            g_gamma: (H,)
            h0:     (HV, K, V)
            retrieve_parent_token: (T,) int or None (chain mode)
        Returns:
            o:      (T, HV, V)
            h_all:  (T, HV, K, V)  # per-step state for sanity comparison
        """
        T, H, K = q.shape
        HV, V = v.shape[1], v.shape[2]
        decay = torch.exp(g_gamma.to(torch.float32))  # (H,)
        # When H != HV (GQA), the kernel maps i_h = i_hv // (HV // H);
        # we replicate that here so the reference uses the same mapping.
        head_per_kv = HV // H
        head_kv_to_h = torch.arange(HV, device=q.device) // max(head_per_kv, 1)

        h_all = torch.zeros(T, HV, K, V, dtype=torch.float32, device=q.device)
        o = torch.zeros(T, HV, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            if t == 0 or retrieve_parent_token is None:
                # Either root (t=0) or linear-chain mode: previous step is t-1.
                h_prev = h0 if t == 0 else h_all[t - 1]
            else:
                parent = int(retrieve_parent_token[t].item())
                h_prev = h0 if parent < 0 else h_all[parent]
            # Apply decay per kv-head (broadcast from head index).
            decay_hv = decay[head_kv_to_h]  # (HV,)
            h_t = decay_hv[:, None, None] * h_prev + (
                k[t].to(torch.float32)[head_kv_to_h][:, :, None]
                * v[t].to(torch.float32)[:, None, :]
            )
            h_all[t] = h_t
            q_t = (q[t].to(torch.float32) * scale)[head_kv_to_h]  # (HV, K)
            o[t] = torch.einsum("hk,hkv->hv", q_t, h_t)
        return o, h_all

    def _allocate_state_pool(self, num_slots, HV, K, V, device, dtype):
        """Single-layer stand-in for ``MambaPool.SpeculativeState.temporal``."""
        return torch.zeros(num_slots, HV, K, V, dtype=dtype, device=device)

    def _allocate_intermediate(self, num_slots, cache_steps, HV, K, V, device, dtype):
        """Single-layer stand-in for the ``intermediate_ssm`` cache."""
        return torch.zeros(
            num_slots, cache_steps, HV, K, V, dtype=dtype, device=device,
        )

    def _build_pyramid_tree(self):
        """Return the ``retrieve_parent_token`` for the 4-token tree::

                  0
                 / \\
                1   2
               /
              3
        """
        return torch.tensor([-1, 0, 0, 1], dtype=torch.int32)

    def _make_inputs(self, T, H, HV, K, V, device, dtype, seed=0):
        torch.manual_seed(seed)
        q = torch.randn(1, T, H, K, dtype=dtype, device=device) * 0.1
        k = torch.randn(1, T, H, K, dtype=dtype, device=device) * 0.1
        v = torch.randn(1, T, HV, V, dtype=dtype, device=device) * 0.1
        g_gamma = torch.full(
            (H,), math.log(0.95), dtype=torch.float32, device=device,
        )
        return q, k, v, g_gamma

    def test_chain_matches_linear_recurrence(self):
        """topk=1 path: retrieve_parent_token=None should equal upstream linear."""
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_simple_gla_update,
        )

        device = torch.device("cuda")
        dtype = torch.float32  # match kernel accumulator precision
        T, H, HV, K, V = 4, 2, 2, 8, 8
        q, k, v, g_gamma = self._make_inputs(T, H, HV, K, V, device, dtype)

        scale = K ** -0.5
        h0_pool = self._allocate_state_pool(1, HV, K, V, device, dtype)
        h0_indices = torch.tensor([0], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

        # Kernel call (chain mode — no tree).
        o = fused_recurrent_simple_gla_update(
            q=q, k=k, v=v,
            g_gamma=g_gamma,
            scale=scale,
            initial_state_source=h0_pool,
            initial_state_indices=h0_indices,
            cu_seqlens=cu_seqlens,
            disable_state_update=True,
            retrieve_parent_token=None,
        ).squeeze(0)  # (T, HV, V)

        # Reference (chain).
        ref_o, _ = self._reference(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            g_gamma, scale, h0_pool[0],
            retrieve_parent_token=None,
        )

        torch.testing.assert_close(o.float(), ref_o.float(), atol=1e-3, rtol=1e-3)

    def test_pyramid_tree_branches_attend_to_ancestors(self):
        """4-token pyramid: tokens 1 and 2 share root state; token 3 branches off 1."""
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_simple_gla_update,
        )

        device = torch.device("cuda")
        dtype = torch.float32
        T, H, HV, K, V = 4, 2, 2, 8, 8
        q, k, v, g_gamma = self._make_inputs(T, H, HV, K, V, device, dtype, seed=1)

        scale = K ** -0.5
        h0_pool = self._allocate_state_pool(1, HV, K, V, device, dtype)
        # Seed the initial state with something nonzero to ensure the
        # USE_INITIAL_STATE branch is exercised.
        h0_pool[0] = torch.randn_like(h0_pool[0]) * 0.05
        h0_indices = torch.tensor([0], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

        cache_steps = T
        intermediate_buf = self._allocate_intermediate(
            1, cache_steps, HV, K, V, device, dtype,
        )
        intermediate_indices = torch.tensor([0], dtype=torch.int32, device=device)

        parent_tree = self._build_pyramid_tree().to(device).unsqueeze(0)  # (1, T)

        o = fused_recurrent_simple_gla_update(
            q=q, k=k, v=v,
            g_gamma=g_gamma,
            scale=scale,
            initial_state_source=h0_pool,
            initial_state_indices=h0_indices,
            cu_seqlens=cu_seqlens,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_buf,
            intermediate_state_indices=intermediate_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=parent_tree,
        ).squeeze(0)

        ref_o, ref_h = self._reference(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            g_gamma, scale, h0_pool[0],
            retrieve_parent_token=parent_tree.squeeze(0),
        )

        torch.testing.assert_close(o.float(), ref_o.float(), atol=1e-3, rtol=1e-3)

        # Sanity: the intermediate buffer should hold each step's h_t.
        # Token 3's reload should have used token 1's state — not the
        # linearly-accumulated h_2. Differs from chain mode iff k_2, v_2
        # actually perturb h.
        torch.testing.assert_close(
            intermediate_buf[0, :T].float(), ref_h.float(),
            atol=1e-3, rtol=1e-3,
        )

    def test_disable_state_update_does_not_mutate_pool(self):
        """The verify path must not commit to the persistent SSM cache."""
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_simple_gla_update,
        )

        device = torch.device("cuda")
        dtype = torch.float32
        T, H, HV, K, V = 4, 2, 2, 8, 8
        q, k, v, g_gamma = self._make_inputs(T, H, HV, K, V, device, dtype, seed=2)

        scale = K ** -0.5
        h0_pool = self._allocate_state_pool(1, HV, K, V, device, dtype)
        h0_pool[0] = torch.randn_like(h0_pool[0]) * 0.05
        h0_pool_before = h0_pool.clone()
        h0_indices = torch.tensor([0], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

        parent_tree = self._build_pyramid_tree().to(device).unsqueeze(0)
        intermediate_buf = self._allocate_intermediate(
            1, T, HV, K, V, device, dtype,
        )
        intermediate_indices = torch.tensor([0], dtype=torch.int32, device=device)

        _ = fused_recurrent_simple_gla_update(
            q=q, k=k, v=v,
            g_gamma=g_gamma,
            scale=scale,
            initial_state_source=h0_pool,
            initial_state_indices=h0_indices,
            cu_seqlens=cu_seqlens,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_buf,
            intermediate_state_indices=intermediate_indices,
            cache_steps=T,
            retrieve_parent_token=parent_tree,
        )

        torch.testing.assert_close(h0_pool, h0_pool_before, atol=0, rtol=0)

    def test_two_request_batch(self):
        """Two requests packed via cu_seqlens, each with its own tree."""
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_simple_gla_update,
        )

        device = torch.device("cuda")
        dtype = torch.float32
        T_per, H, HV, K, V = 4, 2, 2, 8, 8
        # Two requests: same pyramid tree, different inputs.
        q0, k0, v0, g_gamma = self._make_inputs(T_per, H, HV, K, V, device, dtype, seed=3)
        q1, k1, v1, _ = self._make_inputs(T_per, H, HV, K, V, device, dtype, seed=4)

        q = torch.cat([q0, q1], dim=1)  # (1, 2T, H, K)
        k = torch.cat([k0, k1], dim=1)
        v = torch.cat([v0, v1], dim=1)
        scale = K ** -0.5

        h0_pool = self._allocate_state_pool(2, HV, K, V, device, dtype)
        h0_pool[0] = torch.randn_like(h0_pool[0]) * 0.05
        h0_pool[1] = torch.randn_like(h0_pool[1]) * 0.05
        h0_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor(
            [0, T_per, 2 * T_per], dtype=torch.int32, device=device,
        )

        parent_tree_one = self._build_pyramid_tree().to(device)
        parent_tree = torch.stack([parent_tree_one, parent_tree_one], dim=0)  # (2, T)

        intermediate_buf = self._allocate_intermediate(
            2, T_per, HV, K, V, device, dtype,
        )
        intermediate_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)

        o = fused_recurrent_simple_gla_update(
            q=q, k=k, v=v,
            g_gamma=g_gamma,
            scale=scale,
            initial_state_source=h0_pool,
            initial_state_indices=h0_indices,
            cu_seqlens=cu_seqlens,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_buf,
            intermediate_state_indices=intermediate_indices,
            cache_steps=T_per,
            retrieve_parent_token=parent_tree,
        ).squeeze(0)

        # Reference each request independently against its own h0.
        ref_o_0, _ = self._reference(
            q0.squeeze(0), k0.squeeze(0), v0.squeeze(0),
            g_gamma, scale, h0_pool[0],
            retrieve_parent_token=parent_tree_one,
        )
        ref_o_1, _ = self._reference(
            q1.squeeze(0), k1.squeeze(0), v1.squeeze(0),
            g_gamma, scale, h0_pool[1],
            retrieve_parent_token=parent_tree_one,
        )

        torch.testing.assert_close(
            o[:T_per].float(), ref_o_0.float(), atol=1e-3, rtol=1e-3,
        )
        torch.testing.assert_close(
            o[T_per:].float(), ref_o_1.float(), atol=1e-3, rtol=1e-3,
        )


class TestFillRetrieveParentToken(unittest.TestCase):
    """CPU-friendly tests for the pure-torch parent-token fill helper.

    GDN backends rely on ``causal_conv1d_update`` to populate the
    ``retrieve_parent_token`` buffer as a side effect during forward.
    SALA's lightning layers have no conv, so SimpleGLAAttnBackend
    derives the buffer from ``retrieve_next_token`` / ``retrieve_next_sibling``
    via ``_fill_retrieve_parent_token_inplace`` at metadata-init time.
    These tests pin the helper's correctness against hand-computed
    parent trees.

    The helper itself is pure torch (no Triton, no CUDA), but importing
    it goes through the SGLang package which has heavy dependencies
    (tqdm, transformers, ...) that may not exist in a stripped local
    dev mirror. Skip gracefully on ImportError; the tests still run on
    AutoDL / CI where the full SGLang env is installed.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                SimpleGLAAttnBackend,
            )
            cls._fill = SimpleGLAAttnBackend._fill_retrieve_parent_token_inplace
        except ImportError as e:
            raise unittest.SkipTest(
                f"SGLang import unavailable in this env ({e}); helper "
                f"tests will run on AutoDL/CI where deps are installed."
            )

    @staticmethod
    def _reference_parent(next_token: torch.Tensor, next_sibling: torch.Tensor) -> torch.Tensor:
        """Brute-force parent derivation, line-for-line from the conv kernel."""
        bs, N = next_token.shape
        parent = torch.zeros_like(next_token)
        for b in range(bs):
            for t in range(N):
                nt = int(next_token[b, t].item())
                if nt >= 0:
                    parent[b, nt] = t
                ns = int(next_sibling[b, t].item())
                if ns >= 0:
                    parent[b, ns] = parent[b, t]
        return parent

    def test_pyramid_tree(self):
        """The 4-token pyramid:
                  0
                 / \\
                1   2
               /
              3
        next_token = [1, 3, -1, -1], next_sibling = [-1, 2, -1, -1]
        expected: parent = [0, 0, 0, 1]  (parent[0] is "self" by convention).
        """
        nt = torch.tensor([[1, 3, -1, -1]], dtype=torch.int32)
        ns = torch.tensor([[-1, 2, -1, -1]], dtype=torch.int32)
        parent = torch.full((1, 4), -99, dtype=torch.int32)  # poisoned
        self._fill(parent, nt, ns)
        torch.testing.assert_close(
            parent, torch.tensor([[0, 0, 0, 1]], dtype=torch.int32)
        )

    def test_linear_chain(self):
        """4-token chain: 0 -> 1 -> 2 -> 3.

        Each node has exactly one child, no siblings.
        next_token = [1, 2, 3, -1], next_sibling = [-1, -1, -1, -1]
        expected: parent = [0, 0, 1, 2]
        """
        nt = torch.tensor([[1, 2, 3, -1]], dtype=torch.int32)
        ns = torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32)
        parent = torch.full((1, 4), -99, dtype=torch.int32)
        self._fill(parent, nt, ns)
        torch.testing.assert_close(
            parent, torch.tensor([[0, 0, 1, 2]], dtype=torch.int32)
        )

    def test_full_binary_tree_depth2(self):
        """7-token full binary tree of depth 2.

                  0
                 / \\
                1   2
               / \\ / \\
              3  4 5  6
        next_token   = [1, 3, 5, -1, -1, -1, -1]
        next_sibling = [-1, 2, -1, 4, -1, 6, -1]
        expected parent = [0, 0, 0, 1, 1, 2, 2]
        """
        nt = torch.tensor([[1, 3, 5, -1, -1, -1, -1]], dtype=torch.int32)
        ns = torch.tensor([[-1, 2, -1, 4, -1, 6, -1]], dtype=torch.int32)
        parent = torch.zeros(1, 7, dtype=torch.int32)
        self._fill(parent, nt, ns)
        torch.testing.assert_close(
            parent, torch.tensor([[0, 0, 0, 1, 1, 2, 2]], dtype=torch.int32)
        )

    def test_batched_mixed_trees(self):
        """Three batches with different topologies, run as one fill call."""
        nt = torch.tensor(
            [
                [1, 3, -1, -1],   # pyramid
                [1, 2, 3, -1],    # chain
                [1, -1, -1, -1],  # depth-1 with single child
            ],
            dtype=torch.int32,
        )
        ns = torch.tensor(
            [
                [-1, 2, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        parent = torch.full((3, 4), -99, dtype=torch.int32)
        self._fill(parent, nt, ns)
        expected = torch.tensor(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 2],
                [0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(parent, expected)

    def test_matches_brute_force_reference(self):
        """Fuzz against a brute-force reference on hand-built valid trees."""
        nt = torch.tensor(
            [
                [1, 4, -1, -1, 6, -1, -1, -1],
                [1, 3, -1, 5, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        ns = torch.tensor(
            [
                [-1, 2, 3, -1, 5, -1, 7, -1],
                [-1, 2, -1, 4, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        parent = torch.zeros(2, 8, dtype=torch.int32)
        self._fill(parent, nt, ns)
        expected = self._reference_parent(nt, ns)
        torch.testing.assert_close(parent, expected)


if __name__ == "__main__":
    unittest.main()
