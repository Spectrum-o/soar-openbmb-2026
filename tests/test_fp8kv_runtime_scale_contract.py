#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_fp8kv_runtime_scale_contract",
    REPO_ROOT / "tools" / "audit_fp8kv_runtime_scale_contract.py",
)
audit_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_tool
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_tool)


class TestFp8KvRuntimeScaleContract(unittest.TestCase):
    def test_repo_audit_finds_current_hybrid_contract(self):
        checks = audit_tool.audit_sglang_repo(REPO_ROOT)
        by_label = {check.label: check for check in checks}

        self.assertTrue(by_label["docs say FP8 KV scale is scalar per-tensor"].passed)
        self.assertTrue(by_label["BaseKVCacheMethod keeps scalar fallback parameters"].passed)
        self.assertTrue(by_label["BaseKVCacheMethod preserves optional per-head checkpoint scales"].passed)
        self.assertTrue(by_label["MHA KV pool can broadcast per-head scale before FP8 cast"].passed)
        self.assertTrue(by_label["MiniCPM backend wires Q/output per-head bridge"].passed)
        self.assertTrue(by_label["JSON quantization schema stores one float per layer"].passed)
        self.assertTrue(by_label["MiniCPM FlashInfer wrapper receives float scale side channel"].passed)

    def test_flashinfer_audit_distinguishes_single_prefill_from_paged_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "flashinfer"
            package.mkdir()
            (package / "prefill.py").write_text(
                """
def single_prefill_with_kv_cache(
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
):
    '''scale_k : Optional[torch.Tensor]
    The scale tensor for key, per-head quantization with shape: [num_kv_heads].
    scale_v : Optional[torch.Tensor]
    The scale tensor for value, per-head quantization with shape: [num_kv_heads].
    '''

class BatchPrefillWithPagedKVCacheWrapper:
    def run(self, k_scale: Optional[float] = None, v_scale: Optional[float] = None):
        run_args = [
            None,  # scale_k
            None,  # scale_v
        ]
""",
                encoding="utf-8",
            )
            (package / "decode.py").write_text(
                """
class BatchDecodeWithPagedKVCacheWrapper:
    def run(self, k_scale: Optional[float] = None, v_scale: Optional[float] = None):
        run_args = [
            None,  # scale_k
            None,  # scale_v
        ]
""",
                encoding="utf-8",
            )

            checks = audit_tool.audit_flashinfer(root)

        self.assertTrue(all(check.passed for check in checks))


if __name__ == "__main__":
    unittest.main()
