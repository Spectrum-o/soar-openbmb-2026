from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import recalibrate_gptq_w4_scales as recal  # noqa: E402
import dequant_one_tensor as dequant  # noqa: E402


MODULE = "model.layers.0.mlp.gate_proj"


def pack_rows_uint4(values: torch.Tensor) -> torch.Tensor:
    """Pack [in, out] uint4 values into GPTQ qweight [in/8, out]."""
    in_features, out_features = values.shape
    assert in_features % 8 == 0
    out = torch.zeros((in_features // 8, out_features), dtype=torch.int32)
    for i in range(in_features):
        out[i // 8].bitwise_or_((values[i].to(torch.int32) & 0xF) << ((i % 8) * 4))
    return out


def pack_qzeros(qzero: int, groups: int, out_features: int) -> torch.Tensor:
    assert out_features % 8 == 0
    out = torch.zeros((groups, out_features // 8), dtype=torch.int32)
    packed = 0
    for j in range(8):
        packed |= (qzero & 0xF) << (j * 4)
    if packed >= 2**31:
        packed -= 2**32
    out.fill_(packed)
    return out


def write_fixture(root: Path) -> tuple[Path, Path, torch.Tensor]:
    artifact = root / "artifact"
    base = root / "base"
    artifact.mkdir()
    base.mkdir()

    in_features = 16
    out_features = 8
    group_size = 8
    groups = in_features // group_size

    q_centered = torch.tensor(
        [
            [-3, -2, -1, 0, 1, 2, 3, -3],
            [-2, -1, 0, 1, 2, 3, -3, -2],
            [-1, 0, 1, 2, 3, -3, -2, -1],
            [0, 1, 2, 3, -3, -2, -1, 0],
            [1, 2, 3, -3, -2, -1, 0, 1],
            [2, 3, -3, -2, -1, 0, 1, 2],
            [3, -3, -2, -1, 0, 1, 2, 3],
            [-3, -2, -1, 0, 1, 2, 3, -3],
            [-1, -2, -3, 0, 1, 2, 3, -1],
            [-2, -3, 0, 1, 2, 3, -1, -2],
            [-3, 0, 1, 2, 3, -1, -2, -3],
            [0, 1, 2, 3, -1, -2, -3, 0],
            [1, 2, 3, -1, -2, -3, 0, 1],
            [2, 3, -1, -2, -3, 0, 1, 2],
            [3, -1, -2, -3, 0, 1, 2, 3],
            [-1, -2, -3, 0, 1, 2, 3, -1],
        ],
        dtype=torch.float32,
    )
    q_unsigned = (q_centered + 8).to(torch.int32)
    true_scales = torch.tensor(
        [
            [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
            [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
        ],
        dtype=torch.float32,
    )
    old_scales = (true_scales * 0.5).to(torch.float16)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    ref = q_centered.clone()
    ref[:group_size] *= true_scales[0].unsqueeze(0)
    ref[group_size:] *= true_scales[1].unsqueeze(0)

    save_file(
        {
            f"{MODULE}.qweight": pack_rows_uint4(q_unsigned),
            f"{MODULE}.qzeros": pack_qzeros(8, groups, out_features),
            f"{MODULE}.scales": old_scales,
            f"{MODULE}.g_idx": g_idx,
        },
        str(artifact / "model.safetensors"),
    )
    save_file(
        {f"{MODULE}.weight": ref.T.contiguous().to(torch.bfloat16)},
        str(base / "model.safetensors"),
    )
    return artifact, base, true_scales


class TestRecalibrateGptqW4Scales(unittest.TestCase):
    def test_signed_marlin_qzeros_unpack_to_eight(self):
        qzeros = pack_qzeros(8, groups=1, out_features=8)
        self.assertLess(int(qzeros[0, 0].item()), 0)
        unpacked = recal.unpack_qzeros_row(qzeros, group_idx=0, out_features=8)
        self.assertEqual(unpacked.tolist(), [8.0] * 8)

    def test_recalibration_formula_matches_dequant_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, base, _true_scales = write_fixture(Path(tmp))
            qweight = recal.load_tensor(artifact, f"{MODULE}.qweight")
            qzeros = recal.load_tensor(artifact, f"{MODULE}.qzeros")
            scales = recal.load_tensor(artifact, f"{MODULE}.scales")
            g_idx = recal.load_tensor(artifact, f"{MODULE}.g_idx")

            expected, indices = dequant.dequant_gptq_sym_uint4b8(
                qweight, qzeros, scales, g_idx
            )
            rows = torch.arange(qweight.shape[0] * 8, dtype=torch.int64)
            q_unsigned = recal.unpack_qweight_rows(qweight, rows)
            qzero = torch.stack(
                [
                    recal.unpack_qzeros_row(qzeros, int(g_idx[i].item()), qweight.shape[1])
                    for i in rows
                ]
            )
            actual = (q_unsigned - qzero) * scales[g_idx.to(torch.int64)]

            self.assertTrue(torch.equal(indices, rows))
            self.assertTrue(torch.allclose(actual.to(expected.dtype), expected))

    def test_recalibrate_module_scales_reduces_mse(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, base, true_scales = write_fixture(Path(tmp))
            qweight = recal.load_tensor(artifact, f"{MODULE}.qweight")
            qzeros = recal.load_tensor(artifact, f"{MODULE}.qzeros")
            scales = recal.load_tensor(artifact, f"{MODULE}.scales")
            g_idx = recal.load_tensor(artifact, f"{MODULE}.g_idx")
            bf16 = recal.load_tensor(base, f"{MODULE}.weight")

            new_scales, old_mse, new_mse = recal.recalibrate_module_scales(
                qweight, qzeros, scales, g_idx, bf16, eps=1e-12
            )

            self.assertLess(new_mse, old_mse * 0.01)
            self.assertTrue(torch.allclose(new_scales.float(), true_scales, atol=2e-3))

    def test_run_rewrites_only_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, base, true_scales = write_fixture(Path(tmp))
            rc = recal.run(
                type(
                    "Args",
                    (),
                    {
                        "artifact": str(artifact),
                        "base": str(base),
                        "module_regex": "",
                        "limit": None,
                        "report": "",
                        "dry_run": False,
                        "eps": 1e-12,
                    },
                )()
            )
            self.assertEqual(rc, 0)
            state = load_file(str(artifact / "model.safetensors"))
            self.assertTrue(
                torch.allclose(state[f"{MODULE}.scales"].float(), true_scales, atol=2e-3)
            )
            self.assertIn(f"{MODULE}.qweight", state)
            self.assertIn(f"{MODULE}.qzeros", state)


if __name__ == "__main__":
    unittest.main()
