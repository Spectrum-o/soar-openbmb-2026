#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="${OUT_DIR:-/root/autodl-fs/zyn/server_state}"
mkdir -p "${OUT_DIR}" 2>/dev/null || OUT_DIR="${PWD}"
REPORT="${REPORT:-${OUT_DIR}/server_state_$(date '+%Y%m%d_%H%M%S').md}"

append_cmd() {
    local title="$1"
    local cmd="$2"
    {
        printf '\n## %s\n\n' "${title}"
        printf '```bash\n$ %s\n' "${cmd}"
        bash -lc "${cmd}" 2>&1
        printf '```\n'
    } >> "${REPORT}"
}

{
    printf '# AutoDL Server State\n\n'
    printf -- '- generated_at: `%s`\n' "$(date '+%F %T %Z')"
    printf -- '- hostname: `%s`\n' "$(hostname 2>/dev/null || true)"
    printf -- '- user: `%s`\n' "$(whoami 2>/dev/null || true)"
    printf -- '- pwd: `%s`\n' "${PWD}"
    printf -- '- note: read-only audit; no cleanup or process stop was performed\n'
} > "${REPORT}"

append_cmd "System" "uname -a; cat /etc/os-release 2>/dev/null | sed -n '1,12p'"
append_cmd "Important Paths" "for p in /root/autodl-tmp /root/autodl-tmp/zyn /root/autodl-fs /root/autodl-fs/zyn /root/autodl-fs/models /root/autodl-fs/models/OpenBMB/MiniCPM-SALA /root/autodl-fs/models/MiniCPM-SALA-GPTQ-W4A16-probe /root/autodl-fs/models/MiniCPM-SALA-GPTQ-W4A16-probe.partial_20260513; do if [ -e \"\$p\" ]; then echo \"--- \$p\"; ls -ld \"\$p\"; du -sh \"\$p\" 2>/dev/null || true; fi; done"
append_cmd "Disk" "df -h / /root/autodl-tmp /root/autodl-fs 2>/dev/null || df -h"
append_cmd "Top Disk Usage" "for p in /root/autodl-tmp/zyn /root/autodl-fs/zyn /root/autodl-fs/models; do [ -d \"\$p\" ] && { echo \"--- \$p\"; du -h --max-depth=2 \"\$p\" 2>/dev/null | sort -h | tail -80; }; done"
append_cmd "GPU" "nvidia-smi 2>&1 || true"
append_cmd "CUDA And Compilers" "which nvcc 2>/dev/null && nvcc --version || true; which gcc 2>/dev/null && gcc --version | head -1 || true; which g++ 2>/dev/null && g++ --version | head -1 || true"
append_cmd "Listening Ports" "ss -ltnp 2>/dev/null | sed -n '1,120p' || netstat -ltnp 2>/dev/null | sed -n '1,120p' || true"
append_cmd "Related Processes" "ps -eo pid,ppid,etime,stat,cmd | grep -E 'sglang\\.launch_server|bench_serving|quantize_gptqmodel|prepare_model|gptqmodel|python.*sglang' | grep -v grep || true"
append_cmd "Screen And Tmux" "screen -ls 2>&1 || true; tmux ls 2>&1 || true"
append_cmd "Repo Candidates" "for p in /root/autodl-tmp/zyn/sglang /root/autodl-fs/zyn/sglang /root/autodl-tmp/zyn/SGLang-MiniCPM-SALA /root/autodl-fs/zyn/SGLang-MiniCPM-SALA; do if [ -d \"\$p\" ]; then echo \"--- \$p\"; git -C \"\$p\" status --short --branch 2>&1 || true; git -C \"\$p\" log --oneline --decorate -5 2>&1 || true; ls -la \"\$p\" | sed -n '1,30p'; fi; done"
append_cmd "Python And Venvs" "for py in python3 /root/autodl-tmp/zyn/sglang/sglang_minicpm_sala_env/bin/python /root/autodl-tmp/zyn/gptq_env/bin/python /root/autodl-tmp/zyn/gptq_env_20260520/bin/python; do if command -v \"\$py\" >/dev/null 2>&1 || [ -x \"\$py\" ]; then echo \"--- \$py\"; \"\$py\" - <<'PY'\nimport importlib.metadata as m, sys\nprint('exe', sys.executable)\nprint('version', sys.version.split()[0])\nfor pkg in ['torch','transformers','sglang','flash-attn','gptqmodel','huggingface-hub','accelerate','safetensors']:\n    try: print(pkg, m.version(pkg))\n    except Exception: print(pkg, 'NOT_INSTALLED')\nPY\nfi; done"
append_cmd "Model File Heads" "for p in /root/autodl-fs/models/OpenBMB/MiniCPM-SALA /root/autodl-fs/models/MiniCPM-SALA-GPTQ-W4A16-probe; do if [ -d \"\$p\" ]; then echo \"--- \$p\"; find \"\$p\" -maxdepth 1 -type f -printf '%f %s\\n' 2>/dev/null | sort | sed -n '1,80p'; fi; done"
append_cmd "Network Mirror Checks" "for u in https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.aliyun.com/pypi/simple https://hf-mirror.com https://pypi.org/simple; do echo \"--- \$u\"; timeout 8 curl -I -L \"\$u\" 2>&1 | sed -n '1,12p' || true; done"

printf 'Wrote %s\n' "${REPORT}"
