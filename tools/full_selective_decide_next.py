#!/usr/bin/env python3
"""Decide the next full-W4A16 selective-BF16 package to submit.

The SOAR platform score is relative and drifts, so this helper uses concrete
platform metrics: acc_ori and benchmark_duration.{S1,S8,Smax}. It does not call
the platform API. Paste the latest Score JSON on stdin, or pass metrics via
flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    key: str
    tarball: str
    variant: str
    strategy: str
    when: str
    risk: str
    priority: int
    expected_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    candidate: Candidate | None = None


@dataclass(frozen=True)
class PlatformStatus:
    status: str
    message: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        key="last7attn",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last7attn_stalephys_20260528_1740.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last7attn",
        strategy="last7-lightning / attn",
        when="platform-proven parent; superseded by last7attn_rmsopfusion for score",
        risk="platform-proven: final_score=20.86, acc_ori=78.71, S1=650.93",
        priority=10,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last6attn",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last6attn_stalephys_20260528_2200.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last6attn",
        strategy="last6-lightning / attn",
        when="current next speed probe after last7attn reached the best known final_score",
        risk="complete-attn axis, one step more aggressive than last7attn",
        priority=20,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last6-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last7attn_rmsopfusion",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_20260529_1035.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion",
        strategy="last7-lightning / attn + RMSNorm residual-delay opfusion",
        when="platform-proven current best; submit only if you need a rerun",
        risk="platform-proven: final_score=21.73, acc_ori=79.87, S1=650.28; keeps RoPE fp32 upcast",
        priority=25,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last7attn_rmsopfusion_calib600_multi120",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib600_multi120_20260529_1420.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib600_multi120",
        strategy="last7-lightning / attn + RMSNorm opfusion + NUM_CALIB=600 multi-adaptive",
        when="do not submit: platform prepare_model OOM at 83GB during GPTQModel replay",
        risk="failed on 2026-05-29 13:56 with torch.OutOfMemoryError in FLA chunk_simple_gla",
        priority=27,
        expected_lines=(
            'NUM_CALIB="${NUM_CALIB:-600}"',
            'CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"',
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last7attn_rmsopfusion_calib300_multi120",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120_20260529_1405.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_multi120",
        strategy="last7-lightning / attn + RMSNorm opfusion + NUM_CALIB=300 multi-adaptive",
        when="current quality probe after calib600 OOM: moderate GPTQ calibration without changing runtime",
        risk="single-variable calibration change on current best; lower memory peak than calib600, still may lengthen prepare_model",
        priority=28,
        expected_lines=(
            'NUM_CALIB="${NUM_CALIB:-300}"',
            'CALIB_WINDOW_MODE="${CALIB_WINDOW_MODE:-multi-adaptive}"',
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last5attn",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last5attn_stalephys_20260528_2230.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last5attn",
        strategy="last5-lightning / attn",
        when="submit only if last6attn clears the 80 acc_ori gate",
        risk="aggressive complete-attn axis; likely below gate if last6attn is weak",
        priority=30,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last5-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last6attn_g64",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last6attn_g64_stalephys_20260528_2355.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last6attn_g64",
        strategy="last6-lightning / attn / group_size=64",
        when="quality fallback if last6attn g128 lands close but below the 80 acc_ori gate",
        risk="same selective layer set as last6attn, slightly slower/larger due to g64 scales",
        priority=32,
        expected_lines=(
            'EXTRA_ARGS+=(--no-offload-disk --group-size "${GROUP_SIZE:-64}")',
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last6-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last5attn_g64",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last5attn_g64_stalephys_20260529_0005.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last5attn_g64",
        strategy="last5-lightning / attn / group_size=64",
        when="quality fallback if last5attn g128 lands close but below the 80 acc_ori gate",
        risk="same selective layer set as last5attn, slightly slower/larger due to g64 scales",
        priority=34,
        expected_lines=(
            'EXTRA_ARGS+=(--no-offload-disk --group-size "${GROUP_SIZE:-64}")',
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last5-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last4attn",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last4attn_stalephys_20260528_2315.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last4attn",
        strategy="last4-lightning / attn",
        when="submit only if last5attn clears the 80 acc_ori gate",
        risk="very aggressive complete-attn axis; likely below gate unless last5attn has margin",
        priority=35,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last4-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last4attn_g64",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last4attn_g64_stalephys_20260529_0015.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last4attn_g64",
        strategy="last4-lightning / attn / group_size=64",
        when="quality fallback if last4attn g128 lands close but below the 80 acc_ori gate",
        risk="same selective layer set as last4attn, slightly slower/larger due to g64 scales",
        priority=36,
        expected_lines=(
            'EXTRA_ARGS+=(--no-offload-disk --group-size "${GROUP_SIZE:-64}")',
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last4-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last8qkv_last7oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last8qkv_last7oproj_stalephys_20260528_2300.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last8qkv_last7oproj",
        strategy="last8-lightning / qkv + last7-lightning / o_proj",
        when="submit if last7attn fails gate but you want the closest fallback below proven last8-attn",
        risk="most conservative middle path; only one late o_proj stays W4A16",
        priority=35,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last8-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
            'FULL_SELECTIVE_EXTRA_LAYERS="${FULL_SELECTIVE_EXTRA_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_EXTRA_MODULES="${FULL_SELECTIVE_EXTRA_MODULES:-o_proj}"',
        ),
    ),
    Candidate(
        key="last7attn_g64",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last7attn_g64_stalephys_20260528_2345.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last7attn_g64",
        strategy="last7-lightning / attn / group_size=64",
        when="quality fallback if last7attn g128 lands close but below the 80 acc_ori gate",
        risk="same selective layer set as last7attn, slightly slower/larger due to g64 scales",
        priority=37,
        expected_lines=(
            'EXTRA_ARGS+=(--no-offload-disk --group-size "${GROUP_SIZE:-64}")',
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
    Candidate(
        key="last8qkv_last6oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last8qkv_last6oproj_stalephys_20260528_2130.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last8qkv_last6oproj",
        strategy="last8-lightning / qkv + last6-lightning / o_proj",
        when="submit after complete-attn axis drops below gate, or if preserving accuracy is preferred",
        risk="middle path; keeps most late o_proj BF16",
        priority=40,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last8-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
            'FULL_SELECTIVE_EXTRA_LAYERS="${FULL_SELECTIVE_EXTRA_LAYERS:-last6-lightning}"',
            'FULL_SELECTIVE_EXTRA_MODULES="${FULL_SELECTIVE_EXTRA_MODULES:-o_proj}"',
        ),
    ),
    Candidate(
        key="last7qkv_last6oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last7qkv_last6oproj_stalephys_20260528_2300.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last7qkv_last6oproj",
        strategy="last7-lightning / qkv + last6-lightning / o_proj",
        when="submit if last7attn clears gate but last6attn fails; boundary between those two complete-attn points",
        risk="fine-grained middle point between last7-attn and last6-attn",
        priority=44,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
            'FULL_SELECTIVE_EXTRA_LAYERS="${FULL_SELECTIVE_EXTRA_LAYERS:-last6-lightning}"',
            'FULL_SELECTIVE_EXTRA_MODULES="${FULL_SELECTIVE_EXTRA_MODULES:-o_proj}"',
        ),
    ),
    Candidate(
        key="last6qkv_last5oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last6qkv_last5oproj_stalephys_20260528_2315.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last6qkv_last5oproj",
        strategy="last6-lightning / qkv + last5-lightning / o_proj",
        when="submit if last6attn clears gate but last5attn fails; boundary between those two complete-attn points",
        risk="fine-grained middle point between last6-attn and last5-attn",
        priority=45,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last6-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
            'FULL_SELECTIVE_EXTRA_LAYERS="${FULL_SELECTIVE_EXTRA_LAYERS:-last5-lightning}"',
            'FULL_SELECTIVE_EXTRA_MODULES="${FULL_SELECTIVE_EXTRA_MODULES:-o_proj}"',
        ),
    ),
    Candidate(
        key="last5qkv_last4oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last5qkv_last4oproj_stalephys_20260528_2315.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last5qkv_last4oproj",
        strategy="last5-lightning / qkv + last4-lightning / o_proj",
        when="submit if last5attn clears gate but last4attn fails; boundary between those two complete-attn points",
        risk="aggressive fine-grained middle point",
        priority=46,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last5-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
            'FULL_SELECTIVE_EXTRA_LAYERS="${FULL_SELECTIVE_EXTRA_LAYERS:-last4-lightning}"',
            'FULL_SELECTIVE_EXTRA_MODULES="${FULL_SELECTIVE_EXTRA_MODULES:-o_proj}"',
        ),
    ),
    Candidate(
        key="last8qkv_last4oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last8qkv_last4oproj_stalephys_20260528_2100.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last8qkv_last4oproj",
        strategy="last8-lightning / qkv + last4-lightning / o_proj",
        when="submit if last8qkv_last6oproj clears the gate but is still too slow",
        risk="middle path; less o_proj safety",
        priority=57,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last8-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
            'FULL_SELECTIVE_EXTRA_LAYERS="${FULL_SELECTIVE_EXTRA_LAYERS:-last4-lightning}"',
            'FULL_SELECTIVE_EXTRA_MODULES="${FULL_SELECTIVE_EXTRA_MODULES:-o_proj}"',
        ),
    ),
    Candidate(
        key="last8qkv_last2oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last8qkv_last2oproj_stalephys_20260528_2100.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last8qkv_last2oproj",
        strategy="last8-lightning / qkv + last2-lightning / o_proj",
        when="submit if last4oproj clears gate and more speed is needed",
        risk="aggressive middle path",
        priority=60,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last8-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
            'FULL_SELECTIVE_EXTRA_LAYERS="${FULL_SELECTIVE_EXTRA_LAYERS:-last2-lightning}"',
            'FULL_SELECTIVE_EXTRA_MODULES="${FULL_SELECTIVE_EXTRA_MODULES:-o_proj}"',
        ),
    ),
    Candidate(
        key="last6qkv",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last6qkv_stalephys_20260528_2330.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last6qkv",
        strategy="last6-lightning / qkv",
        when="qkv-only probe after stronger last6-ish boundary candidates clear the gate",
        risk="high accuracy risk; o_proj stays W4A16",
        priority=61,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last6-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
        ),
    ),
    Candidate(
        key="last5qkv",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last5qkv_stalephys_20260528_2330.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last5qkv",
        strategy="last5-lightning / qkv",
        when="qkv-only probe if last5/last4 boundary still clears the gate",
        risk="very high accuracy risk; o_proj stays W4A16",
        priority=62,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last5-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
        ),
    ),
    Candidate(
        key="last4qkv",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last4qkv_stalephys_20260528_2330.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last4qkv",
        strategy="last4-lightning / qkv",
        when="most aggressive qkv-only probe; submit only if last4attn clears gate",
        risk="highest accuracy risk among primary packaged probes",
        priority=63,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last4-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
        ),
    ),
    Candidate(
        key="last8qkv",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last8qkv_stalephys_20260528_1740.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last8qkv",
        strategy="last8-lightning / qkv",
        when="submit only if qkv-only risk is acceptable or middle paths are too slow",
        risk="high accuracy risk if o_proj drove recovery",
        priority=70,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last8-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
        ),
    ),
    Candidate(
        key="last7qkv",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last7qkv_stalephys_20260528_2000.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last7qkv",
        strategy="last7-lightning / qkv",
        when="most aggressive speed probe after qkv-only works",
        risk="highest useful accuracy risk",
        priority=80,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last7-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-qkv}"',
        ),
    ),
    Candidate(
        key="last8oproj",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last8oproj_stalephys_20260528_2030.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16_last8oproj",
        strategy="last8-lightning / o_proj",
        when="ablation only; not a primary submit unless testing o_proj importance",
        risk="high gate risk; qkv stays W4A16",
        priority=90,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last8-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-o_proj}"',
        ),
    ),
)


PROVEN_FALLBACKS: dict[str, Candidate] = {
    "last8attn": Candidate(
        key="last8attn",
        tarball="soar_gptqmodel_full_w4a16_selective_bf16_last8attn_stalephys_20260528_1305.tar.gz",
        variant="submission_gptqmodel_full_w4a16_selective_bf16",
        strategy="last8-lightning / attn",
        when="platform-proven fallback; do not resubmit unless you need a rerun",
        risk="already platform-proven: acc_ori=80.51, S1=708.97",
        priority=0,
        expected_lines=(
            'FULL_SELECTIVE_LAYERS="${FULL_SELECTIVE_LAYERS:-last8-lightning}"',
            'FULL_SELECTIVE_MODULES="${FULL_SELECTIVE_MODULES:-attn}"',
        ),
    ),
}


EXPECTED_MD5_BY_KEY: dict[str, str] = {
    "last7attn": "fade038a827731ca6bf3efe227d25b3b",
    "last6attn": "2891556e51e0aba9114ea416eb6964e7",
    "last7attn_rmsopfusion": "137d831db8cbcd684662b35e231a4fc3",
    "last7attn_rmsopfusion_calib600_multi120": "45a28194e3daa94e6804b596d8f8bbbf",
    "last7attn_rmsopfusion_calib300_multi120": "4b131944f190b56d11d6d627ba07b483",
    "last5attn": "e15ef98c9b34229902d506fa7c366f13",
    "last6attn_g64": "45a80d0181478da91465abdf8be802b8",
    "last5attn_g64": "0cd412c07b21a8f36ae25f0644c5dc52",
    "last4attn": "407250a95c79b93c23c8573643a1b60f",
    "last4attn_g64": "1848be336d717c9a21c0900980df9c8a",
    "last8qkv_last7oproj": "72d908a9a3ac7564f9356c1769331c0f",
    "last7attn_g64": "6ee31016e67ed22c690ae5f8aec5d9b0",
    "last8qkv_last6oproj": "c8ab10b0ce9ffa29bdb7ebaad1660c00",
    "last7qkv_last6oproj": "f69b330b3aa7386f8b80fc9f9faf6c91",
    "last6qkv_last5oproj": "29a346f5b37314fda78538f475d254d2",
    "last5qkv_last4oproj": "0b3efa4cc8fa79d09d64fc66ea9d5adc",
    "last8qkv_last4oproj": "7513e0bc4fca8449a14f9f1bd12acf83",
    "last8qkv_last2oproj": "5c3c1e16b34a6d170bc219466cc77c9e",
    "last6qkv": "eb0c1762c5f6153d31378a84cd769c9a",
    "last5qkv": "b247230ad6eccf63633ee277639cf631",
    "last4qkv": "fd1feab118931e6f50be32cb3f69bd81",
    "last8qkv": "6264ce0e9931c1933e6da201ffc16c24",
    "last7qkv": "12ac3983d9ee1ebc748a42680bc63fca",
    "last8oproj": "769c2496e6fac7e783f395851c4f94c3",
    "last8attn": "858c33d2222a4273cae4b751ce74f07f",
}


KNOWN_RESULTS: dict[str, dict[str, float]] = {
    "full_w4a16": {"acc_ori": 78.27, "S1": 621.91, "S8": 1002.06, "Smax": 2315.78},
    "last8attn": {"acc_ori": 80.51, "S1": 708.97, "S8": 1071.87, "Smax": 2376.48},
    "last7attn": {
        "acc": 98.39,
        "acc_ori": 78.71,
        "final_score": 20.86,
        "S1": 650.93,
        "S8": 1024.02,
        "Smax": 2330.54,
    },
    "last7attn_rmsopfusion": {
        "acc": 99.83,
        "acc_ori": 79.87,
        "final_score": 21.73,
        "S1": 650.28,
        "S8": 1023.59,
        "Smax": 2334.43,
    },
    "chunk32k_safe": {"acc_ori": 80.31, "S1": 720.76, "S8": 1081.21, "Smax": 2386.35},
    "v5j_dtype_bf16": {"acc_ori": 82.18, "S1": 717.76, "S8": 1067.43, "Smax": 2343.38},
}


MIN_ACC_ORI_FOR_SPEED_PROBE = 78.0
PASS_FINAL_SCORE = 20.8


PASS_NEXT: dict[str, str] = {
    "last7attn": "last6attn",
    "last7attn_rmsopfusion": "last7attn_rmsopfusion_calib300_multi120",
    "last7attn_rmsopfusion_calib300_multi120": "last6attn",
    "last7attn_rmsopfusion_calib600_multi120": "last6attn",
    "last6attn": "last5attn",
    "last6attn_g64": "last5attn",
    "last5attn": "last4attn",
    "last5attn_g64": "last4attn",
    "last4attn": "last4qkv",
    "last4attn_g64": "last4qkv",
    # Conservative branch below proven last8-attn: keep qkv safety first,
    # then reduce o_proj safety, then enter qkv-only.
    "last8qkv_last7oproj": "last8qkv_last6oproj",
    "last8qkv_last6oproj": "last8qkv_last4oproj",
    "last8qkv_last4oproj": "last8qkv_last2oproj",
    "last8qkv_last2oproj": "last8qkv",
    # Boundary branch after a complete-attn point fails.
    "last7qkv_last6oproj": "last6qkv_last5oproj",
    "last6qkv_last5oproj": "last5qkv_last4oproj",
    "last5qkv_last4oproj": "last5qkv",
    # Pure qkv-only speed probes.
    "last8qkv": "last7qkv",
    "last7qkv": "last6qkv",
    "last6qkv": "last5qkv",
    "last5qkv": "last4qkv",
}


FAIL_NEXT: dict[str, str | None] = {
    "last7attn": "last7attn_g64",
    "last7attn_rmsopfusion": "last7attn",
    "last7attn_rmsopfusion_calib300_multi120": "last7attn_rmsopfusion",
    "last7attn_rmsopfusion_calib600_multi120": "last7attn_rmsopfusion",
    "last7attn_g64": "last8qkv_last7oproj",
    "last6attn": "last6attn_g64",
    "last6attn_g64": "last7qkv_last6oproj",
    "last5attn": "last5attn_g64",
    "last5attn_g64": "last6qkv_last5oproj",
    "last4attn": "last4attn_g64",
    "last4attn_g64": "last5qkv_last4oproj",
    # If the closest point below proven last8-attn fails, no packaged point is
    # safer than the already-submitted last8-attn result.
    "last8qkv_last7oproj": None,
    "last8qkv_last6oproj": "last8qkv_last7oproj",
    "last8qkv_last4oproj": "last8qkv_last6oproj",
    "last8qkv_last2oproj": "last8qkv_last4oproj",
    "last7qkv_last6oproj": "last7attn",
    "last6qkv_last5oproj": "last7qkv_last6oproj",
    "last5qkv_last4oproj": "last6qkv_last5oproj",
    "last8qkv": "last8qkv_last2oproj",
    "last7qkv": "last8qkv",
    "last6qkv": "last7qkv",
    "last5qkv": "last6qkv",
    "last4qkv": "last5qkv",
    "last8oproj": "last8attn",
}


FAIL_NEXT_IS_FALLBACK: frozenset[str] = frozenset(
    {
        "last7attn_rmsopfusion",
        "last7attn_rmsopfusion_calib300_multi120",
        "last7attn_rmsopfusion_calib600_multi120",
        "last8qkv_last6oproj",
        "last8qkv_last4oproj",
        "last8qkv_last2oproj",
        "last7qkv_last6oproj",
        "last6qkv_last5oproj",
        "last5qkv_last4oproj",
        "last8qkv",
        "last7qkv",
        "last6qkv",
        "last5qkv",
        "last4qkv",
        "last8oproj",
    }
)


QUEUE_ORDER: tuple[str, ...] = (
    "last7attn_rmsopfusion",
    "last7attn_rmsopfusion_calib300_multi120",
    "last7attn",
    "last7attn_g64",
    "last8qkv_last7oproj",
    "last6attn",
    "last6attn_g64",
    "last7qkv_last6oproj",
    "last5attn",
    "last5attn_g64",
    "last6qkv_last5oproj",
    "last4attn",
    "last4attn_g64",
    "last5qkv_last4oproj",
    "last4qkv",
    "last8qkv_last6oproj",
    "last8qkv_last4oproj",
    "last8qkv_last2oproj",
    "last8qkv",
    "last7qkv",
    "last6qkv",
    "last5qkv",
    "last8oproj",
)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def md5_display(cand: Candidate, path: Path) -> str:
    if not path.exists():
        return "MISSING"
    actual = md5_file(path)
    expected = EXPECTED_MD5_BY_KEY.get(cand.key)
    if expected is None:
        return f"{actual} (UNPINNED)"
    if actual != expected:
        return f"{actual} (EXPECTED {expected})"
    return actual


def candidate_by_key(key: str) -> Candidate | None:
    for cand in CANDIDATES:
        if cand.key == key:
            return cand
    return PROVEN_FALLBACKS.get(key)


def sorted_primary_candidates() -> list[Candidate]:
    return sorted(CANDIDATES, key=lambda c: c.priority)


def submit_decision(key: str, reason: str) -> Decision:
    cand = candidate_by_key(key)
    if cand is None:
        raise KeyError(f"unknown candidate key: {key}")
    return Decision(action="submit", reason=reason, candidate=cand)


def stop_decision(reason: str, fallback_key: str | None = None) -> Decision:
    cand = candidate_by_key(fallback_key) if fallback_key else None
    return Decision(action="stop", reason=reason, candidate=cand)


def parse_score_text(text: str) -> dict[str, float]:
    text = text.strip()
    if not text:
        return {}

    # Prefer JSON if present, including logs with a JSON object embedded.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            out: dict[str, float] = {}
            for key in ("acc", "acc_ori", "final_score"):
                if key in obj:
                    out[key] = float(obj[key])
            bench = obj.get("benchmark_duration", {})
            if isinstance(bench, dict):
                for key in ("S1", "S8", "Smax"):
                    if key in bench:
                        out[key] = float(bench[key])
            return out
        except Exception:
            pass

    out = {}
    for key in ("acc", "acc_ori", "final_score", "S1", "S8", "Smax"):
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            rf"\b{re.escape(key)}\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                out[key] = float(m.group(1))
                break
    return out


def parse_platform_status_text(text: str) -> PlatformStatus | None:
    """Extract the last SOAR platform status line from a pasted platform log."""
    matches = list(
        re.finditer(
            r"^\s*\[[^\]\n]+\]\s+\[(?P<status>[A-Z_]+)\]\s*(?P<message>.*)$",
            text,
            flags=re.MULTILINE,
        )
    )
    if not matches:
        return None
    match = matches[-1]
    return PlatformStatus(
        status=match.group("status"),
        message=match.group("message").strip(),
    )


def read_score_input(score_json: str | None, score_file: str | None) -> str:
    if score_json is not None and score_file is not None:
        raise ValueError("use only one of --score-json or --score-file")
    if score_json is not None:
        return score_json
    if score_file is not None:
        if score_file == "-":
            return sys.stdin.read()
        return Path(score_file).read_text(encoding="utf-8")
    return sys.stdin.read()


def print_candidate_table() -> int:
    print("Full-W4A16 selective candidates:")
    for cand in sorted_primary_candidates():
        path = REPO_ROOT / cand.tarball
        md5 = md5_display(cand, path)
        size = f"{path.stat().st_size // (1024 * 1024)}MB" if path.exists() else "-"
        print(f"- {cand.key:22s} {size:>5s} {md5}  {cand.tarball}")
        print(f"  strategy: {cand.strategy}")
        print(f"  use:      {cand.when}")
    return 0


def print_variant_list() -> int:
    for cand in sorted_primary_candidates():
        print(cand.variant)
    return 0


def print_queue_map() -> int:
    """Print the actual pass/fail decision graph used after platform results."""
    print("Full-W4A16 selective decision queue:")
    print(
        "gate: final_score >= 20.8 and acc_ori >= 78.0 => pass; "
        "otherwise acc_ori >= 80.0 => pass"
    )
    print("current next submit: last7attn_rmsopfusion_calib300_multi120")
    print("")
    ordered = list(QUEUE_ORDER)
    for key in sorted((PASS_NEXT.keys() | FAIL_NEXT.keys()) - set(ordered)):
        ordered.append(key)
    for key in ordered:
        cand = candidate_by_key(key)
        if cand is None:
            print(f"- {key}: UNKNOWN")
            continue
        pass_key = PASS_NEXT.get(key)
        fail_key = FAIL_NEXT.get(key)
        pass_text = pass_key if pass_key is not None else "stop"
        if fail_key is None:
            fail_text = "stop -> last8attn fallback"
        elif key in FAIL_NEXT_IS_FALLBACK:
            fail_text = f"stop -> {fail_key} fallback"
        else:
            fail_text = fail_key
        print(f"- {key}: pass -> {pass_text}; fail -> {fail_text}")
        print(f"  strategy: {cand.strategy}")
    return 0


def print_candidate_action(cand: Candidate, action: str = "submit") -> None:
    path = REPO_ROOT / cand.tarball
    label = "submit" if action == "submit" else "fallback"
    print(f"{label}: {cand.tarball}")
    print(f"key:    {cand.key}")
    print(f"md5:    {md5_display(cand, path)}")
    print(f"size:   {path.stat().st_size // (1024 * 1024)}MB" if path.exists() else "size:   MISSING")
    print(f"strategy: {cand.strategy}")
    print(f"risk:     {cand.risk}")


def print_current_submit() -> int:
    cand = candidate_by_key("last7attn_rmsopfusion_calib300_multi120")
    if cand is None:
        raise RuntimeError("last7attn_rmsopfusion_calib300_multi120 candidate is missing")
    print("current next submit:")
    print_candidate_action(cand)
    return 0


def _read_tar_member_text(tf: tarfile.TarFile, name: str) -> str:
    member = tf.getmember(name)
    f = tf.extractfile(member)
    if f is None:
        raise RuntimeError(f"cannot read {name}")
    return f.read().decode("utf-8", errors="replace")


def audit_candidate_tarball(cand: Candidate) -> list[str]:
    """Return audit problems for one submission tarball."""
    problems: list[str] = []
    tar_path = REPO_ROOT / cand.tarball
    if not tar_path.is_file():
        return [f"missing tarball: {cand.tarball}"]

    expected_md5 = EXPECTED_MD5_BY_KEY.get(cand.key)
    if expected_md5 is None:
        problems.append(f"missing expected md5 pin for {cand.key}")
    else:
        actual_md5 = md5_file(tar_path)
        if actual_md5 != expected_md5:
            problems.append(
                f"md5 mismatch for {cand.tarball}: actual {actual_md5}, expected {expected_md5}"
            )

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            names = set(tf.getnames())
            for root_file in ("./prepare_env.sh", "./prepare_model.sh"):
                if root_file not in names:
                    problems.append(f"missing root file {root_file}")

            bad_patterns = (
                "__pycache__",
                ".cache/",
                "flashinfer/0.5.3",
                "fp8kv_platform_prewarm_home",
            )
            bad_paths = [
                name for name in names if any(pattern in name for pattern in bad_patterns)
            ]
            if bad_paths:
                problems.append(f"bad cached path(s): {bad_paths[:3]}")

            if "./prepare_env.sh" in names:
                env_text = _read_tar_member_text(tf, "./prepare_env.sh")
                active_env_text = "\n".join(
                    line for line in env_text.splitlines()
                    if not line.lstrip().startswith("#")
                )
                if "--kv-cache-dtype fp8_" in active_env_text:
                    problems.append("prepare_env.sh contains fp8 KV cache dtype")
                if "--disable-cuda-graph" in active_env_text:
                    problems.append("prepare_env.sh disables CUDA graph")
                for required in (
                    "--attention-backend minicpm_flashinfer",
                    "--chunked-prefill-size 32768",
                    "--max-prefill-tokens 32768",
                    "--mem-fraction-static 0.70",
                    "--quantization gptq_marlin",
                    "--dtype bfloat16",
                ):
                    if required not in active_env_text:
                        problems.append(f"prepare_env.sh missing {required}")

            if "./prepare_model.sh" in names:
                model_text = _read_tar_member_text(tf, "./prepare_model.sh")
                for expected in cand.expected_lines:
                    if expected not in model_text:
                        problems.append(f"prepare_model.sh missing expected default: {expected}")
    except Exception as exc:
        problems.append(f"tar read failed: {exc}")

    return problems


def audit_all_tarballs() -> int:
    failed = False
    audit_candidates = sorted_primary_candidates()
    audit_candidates.extend(
        sorted(PROVEN_FALLBACKS.values(), key=lambda c: (c.priority, c.key))
    )
    for cand in audit_candidates:
        problems = audit_candidate_tarball(cand)
        if problems:
            failed = True
            print(f"[FAIL] {cand.key}: {cand.tarball}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            path = REPO_ROOT / cand.tarball
            print(
                f"[PASS] {cand.key:22s} {path.stat().st_size // (1024 * 1024)}MB "
                f"{md5_display(cand, path)}"
            )
    return 1 if failed else 0


def decide_next(
    last_key: str | None,
    metrics: dict[str, float],
    status: PlatformStatus | None = None,
) -> Decision:
    acc_ori = metrics.get("acc_ori")
    final_score = metrics.get("final_score")
    s1 = metrics.get("S1")

    if not last_key:
        return submit_decision(
            "last7attn",
            "no latest variant specified; start with closest-to-proven last7attn",
        )

    if last_key == "last8attn":
        return submit_decision(
            "last7attn",
            "last8attn is proven; first speed-recovery step is last7attn",
        )

    if acc_ori is None:
        if status is not None:
            if status.status == "FAILED":
                return stop_decision(
                    f"{last_key} failed before a Score JSON was produced ({status.message}); "
                    "do not advance the optimization queue until the failure log is fixed",
                    fallback_key=last_key if candidate_by_key(last_key) else "last8attn",
                )
            if status.status == "INFERENCING":
                return stop_decision(
                    f"{last_key} is still inferencing; wait for SUCCESS Score JSON before choosing the next package",
                    fallback_key=last_key if candidate_by_key(last_key) else "last8attn",
                )
            if status.status in {"PENDING", "PREPARING", "DOWNLOADING"}:
                return stop_decision(
                    f"{last_key} has not reached a scored result yet (latest status: {status.status}); "
                    "wait for SUCCESS Score JSON",
                    fallback_key=last_key if candidate_by_key(last_key) else "last8attn",
                )
        return stop_decision(
            "acc_ori missing; do not advance this queue until the platform Score JSON is available",
            fallback_key=last_key if candidate_by_key(last_key) else "last8attn",
        )

    clears_acc_gate = acc_ori >= 80.0
    clears_score_gate = (
        final_score is not None
        and final_score >= PASS_FINAL_SCORE
        and acc_ori >= MIN_ACC_ORI_FOR_SPEED_PROBE
    )

    if clears_acc_gate or clears_score_gate:
        next_key = PASS_NEXT.get(last_key)
        if next_key is None:
            return stop_decision(
                f"{last_key} cleared the platform score gate, and there is no faster packaged point in this queue",
                fallback_key=last_key,
            )
        gate_reason = (
            f"final_score={final_score:.2f} with acc_ori={acc_ori:.2f}"
            if clears_score_gate and not clears_acc_gate and final_score is not None
            else "the 80 acc_ori gate"
        )
        extra = ""
        if s1 is not None and s1 <= KNOWN_RESULTS["full_w4a16"]["S1"] + 20:
            extra = " Current S1 is already near full-W4A16; stop unless acc margin is large."
        return submit_decision(
            next_key,
            f"{last_key} cleared {gate_reason}; next faster packaged point is {next_key}.{extra}",
        )

    next_key = FAIL_NEXT.get(last_key)
    if next_key is None:
        return stop_decision(
            f"{last_key} failed the 80 gate; no safer untested package remains below proven last8attn",
            fallback_key="last8attn",
        )
    if last_key in FAIL_NEXT_IS_FALLBACK:
        return stop_decision(
            f"{last_key} failed the 80 gate; fall back to the previously safer point {next_key} instead of resubmitting",
            fallback_key=next_key,
        )
    return submit_decision(
        next_key,
        f"{last_key} failed the 80 gate; try the safer/boundary package {next_key}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="List candidate packages, md5, and strategies")
    ap.add_argument("--variants", action="store_true", help="Print primary candidate variant directories, one per line")
    ap.add_argument("--queue", action="store_true", help="Print the actual pass/fail decision graph")
    ap.add_argument("--current", action="store_true", help="Print the current first package to submit")
    ap.add_argument("--audit-tarballs", action="store_true", help="Audit all candidate tarballs for package hazards")
    ap.add_argument("--last", help="Latest submitted candidate key, e.g. last7attn")
    ap.add_argument("--score-json", help="Score JSON string pasted from platform")
    ap.add_argument("--score-file", help="File containing platform Score JSON/log; use '-' for stdin")
    ap.add_argument("--acc-ori", type=float, help="Latest acc_ori")
    ap.add_argument("--s1", type=float, help="Latest benchmark_duration.S1")
    ap.add_argument("--s8", type=float, help="Latest benchmark_duration.S8")
    ap.add_argument("--smax", type=float, help="Latest benchmark_duration.Smax")
    args = ap.parse_args()

    if args.list:
        return print_candidate_table()
    if args.variants:
        return print_variant_list()
    if args.queue:
        return print_queue_map()
    if args.current:
        return print_current_submit()
    if args.audit_tarballs:
        return audit_all_tarballs()

    try:
        text = read_score_input(args.score_json, args.score_file)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    metrics = parse_score_text(text)
    status = parse_platform_status_text(text)
    for key, value in (("acc_ori", args.acc_ori), ("S1", args.s1), ("S8", args.s8), ("Smax", args.smax)):
        if value is not None:
            metrics[key] = value

    decision = decide_next(args.last, metrics, status)
    cand = decision.candidate
    print(f"latest: {args.last or '(none)'}")
    if status is not None:
        status_text = status.status if not status.message else f"{status.status} — {status.message}"
        print(f"platform_status: {status_text}")
    print(f"metrics: {metrics if metrics else '(none)'}")
    print(f"decision: {decision.reason}")
    print(f"action: {decision.action}")
    print("")
    if cand is not None:
        print_candidate_action(cand, decision.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
