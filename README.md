<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

> [!WARNING]
> **Customized Version for MiniCPM-Sala**
>
> This repository is a customized version of SGLang specifically optimized for the **OpenBMB MiniCPM-Sala** model.
> Please **ONLY** use this version for running the MiniCPM-Sala model.
>
> **Environment Dependencies:**
> This project is designed to work in conjunction with the following repositories to form the complete runtime environment:
> 1. **InfLLM V2 CUDA Kernels**: [OpenBMB/infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl/tree/minicpm_sala) (included as submodule, branch: `minicpm_sala`)
> 2. **Sparse Kernels**: [OpenBMB/sparse_kernel](https://github.com/OpenBMB/sparse_kernel) (included as submodule)

--------------------------------------------------------------------------------

[**English**] | [中文](./README_zh.md)

# MiniCPM-SALA Inference Environment Setup

## SOAR Workspace Navigation

This branch is not just the upstream MiniCPM-SALA runtime. It is also the active
SOAR 2026 competition workspace, so the repo contains runtime code plus submission
variants, experiments, and operational tooling.

Start here if you are working on the competition workflow:

- `PROJECT_STRUCTURE.md`: repo map and where things live
- `CLAUDE.md`: current workflow, invariants, and common commands
- `experiments/README.md`: index of strategy docs and runbooks
- `SUBMISSIONS.md`: canonical submission log and hard constraints

Important layout note: root-level `submission_*` directories are intentionally kept
at repo root because the pack/eval/preflight scripts reference them directly.

## Requirements

- CUDA 12.x or higher
- `gcc` / `g++` compiler
- `uv` package manager (script will check)

## Quick Start

### Installation

```bash
# Clone repository
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# One-click installation (creates venv and compiles all dependencies)
bash install_minicpm_sala.sh

# Or specify PyPI mirror
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

The installation script performs the following steps:

1. Creates `sglang_minicpm_sala_env` virtual environment (Python 3.12)
2. Clones dependencies to `3rdparty/` (infllmv2) and initializes submodules (sparse_kernel)
3. Installs MiniCPM-SALA (current repo)
4. Compiles and installs `infllmv2_cuda_impl`
5. Compiles and installs `sparse_kernel`
6. Installs `tilelang` & `flash-linear-attention`

### Usage

```bash
# Activate environment
source sglang_minicpm_sala_env/bin/activate

# Launch Inference Server (Replace MODEL_PATH with actual path)
MODEL_PATH=/path/to/your/model

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse
```

| Parameter | Description |
|-----------|-------------|
| `--trust-remote-code` | Allow custom code in model |
| `--disable-radix-cache` | Disable RadixAttention prefix cache |
| `--attention-backend minicpm_flashinfer` | Use MiniCPM FlashInfer backend |
| `--chunked-prefill-size 8192` | Chunked prefill size |
| `--max-running-requests 32` | Max concurrent requests |
| `--skip-server-warmup` | Skip server warmup |
| `--port 31111` | Server port |
| `--dense-as-sparse` | Use dense-as-sparse mode |

> **Tip:** For best generation quality, we recommend setting `temperature=0.9` when sending requests to the server.

### Tool Calling

To enable tool calling, add `--tool-call-parser minicpm4_xml` when launching the server:

```bash
# Activate environment
source sglang_minicpm_sala_env/bin/activate

# Launch Inference Server with Tool Calling (Replace MODEL_PATH with actual path)
MODEL_PATH=/path/to/your/model

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse \
    --tool-call-parser minicpm4_xml
```

**Example Request:**

```bash
curl -X POST "http://localhost:31111/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm4.6-8b",
    "messages": [{"role": "user", "content": "北京天气怎么样"}],
    "chat_template_kwargs": {"enable_thinking": false},
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
    }]
  }'
```

## Directory Structure

```
sglang/
├── README.md                       # This file
├── README_zh.md                    # Chinese documentation
├── install_minicpm_sala.sh         # Installation script
├── 3rdparty/                       # Dependency sources
│   ├── infllmv2_cuda_impl/         # InfLLM v2 CUDA implementation
│   └── sparse_kernel/              # sparse_kernel_extension
├── python/                         # SGLang source code
└── ...
```

## Manual Installation

If the script doesn't work for you, follow these steps:

```bash
# 0. Ensure uv is installed
pip install uv

# 1. Create venv
uv venv --python 3.12 sglang_minicpm_sala_env
source sglang_minicpm_sala_env/bin/activate

# 2. Install SGLang
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./python[all]

# 3. Compile CUDA Extensions
# (Ensure dependencies are cloned to 3rdparty/)
cd 3rdparty/infllmv2_cuda_impl && python setup.py install && cd ../..
cd 3rdparty/sparse_kernel && python setup.py install && cd ../..

# 4. Install extra deps
uv pip install tilelang flash-linear-attention
```

## Q&A

**Q: CUDA extension compilation failed?**

- Ensure CUDA 12+ is installed (`nvcc --version`).
- Ensure `gcc` / `g++` are available.
- If `CXX` is set to `clang++ -pthread`, manually `export CXX=g++`.
