#!/usr/bin/env bash
# peek - v23 pipeline live status snapshot.
#
# Prints a one-screen summary of what's currently running, GPU state,
# latest log signals (versions, [qzeros-fix], DIAGNOSTIC, FATAL/WARN),
# and partial predictions if any eval is in progress.
#
# Designed to be FAST (~1s) so you can paste-loop it freely, or wrap
# with `watch -n 10 bash scripts/peek.sh` for auto-refresh.
#
# Usage:
#     bash scripts/peek.sh                           # one shot
#     watch -n 5 bash scripts/peek.sh                # refresh every 5s
#     PEEK_LOG_DIR=/some/other/path bash scripts/peek.sh
#     bash scripts/peek.sh --no-color                # plain text (default if stdout not a tty)
#
# Flags:
#     --no-color              Disable ANSI colors regardless of tty.
#     --watch [N]             Self-loop every N seconds (default 10). Ctrl+C to exit.
#     -h, --help              Show this help.
#
# Environment:
#     PEEK_LOG_DIR            Override the AutoDL persistent log dir.
#                             Default: /root/autodl-fs/zyn/logs
#     PEEK_OUTPUTS_DIR        Override the outputs dir.
#                             Default: <repo>/outputs

set -uo pipefail

NO_COLOR=0
WATCH_INTERVAL=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEEK_LOG_DIR="${PEEK_LOG_DIR:-/root/autodl-fs/zyn/logs}"
PEEK_OUTPUTS_DIR="${PEEK_OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
SCRIPT_LOG_DIR="${REPO_ROOT}/scripts/logs"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-color) NO_COLOR=1; shift ;;
        --watch)
            shift
            if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
                WATCH_INTERVAL="$1"; shift
            else
                WATCH_INTERVAL=10
            fi
            ;;
        -h|--help)
            grep -E '^# (Usage|Flags|Environment|    )' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ "$WATCH_INTERVAL" -gt 0 ]; then
    while true; do
        clear 2>/dev/null || true
        bash "$0" --no-color
        echo
        echo "(refresh in ${WATCH_INTERVAL}s — Ctrl+C to exit)"
        sleep "$WATCH_INTERVAL"
    done
    exit 0
fi

# ANSI color helpers
if [ "$NO_COLOR" -eq 1 ] || [ ! -t 1 ]; then
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; RESET=""
else
    BOLD="\033[1m"; DIM="\033[2m"
    GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"
    RESET="\033[0m"
fi

hr() { echo -e "${DIM}--------------------------------------------------------------------${RESET}"; }
hdr() { echo -e "${BOLD}${CYAN}=== $1 ===${RESET}"; }

now=$(date '+%Y-%m-%d %H:%M:%S')
hostname_s=$(hostname 2>/dev/null || echo "?")
echo -e "${BOLD}peek${RESET} ${DIM}${hostname_s} @ ${now}${RESET}"
echo

# ---------------------------------------------------------------------------
# 1. Active processes
# ---------------------------------------------------------------------------
hdr "Processes"
found_any=0
while read -r line; do
    if [ -n "$line" ]; then
        found_any=1
        # Extract pid, %cpu, rss, etime, command
        pid=$(echo "$line" | awk '{print $2}')
        rss=$(echo "$line" | awk '{print $6}')
        # human-readable rss
        rss_h=$(awk -v r="$rss" 'BEGIN{
            r=r*1024;
            if (r > 1024*1024*1024) printf "%.1fG", r/1024/1024/1024;
            else if (r > 1024*1024) printf "%.1fM", r/1024/1024;
            else if (r > 1024) printf "%.0fK", r/1024;
            else printf "%dB", r;
        }')
        etime=$(echo "$line" | awk '{print $10}')
        cmd=$(echo "$line" | awk '{for (i=11; i<=NF; i++) printf "%s ", $i; print ""}' | head -c 80)
        # Heuristic tag
        tag=""
        if echo "$cmd" | grep -q "sglang.launch_server"; then tag="${GREEN}[sglang]${RESET}"
        elif echo "$cmd" | grep -q "quantize_gptqmodel"; then tag="${YELLOW}[gptq-quant]${RESET}"
        elif echo "$cmd" | grep -q "quantize_gptq_rtn"; then tag="${YELLOW}[rtn-quant]${RESET}"
        elif echo "$cmd" | grep -q "eval_model.py"; then tag="${CYAN}[eval]${RESET}"
        elif echo "$cmd" | grep -q "watchdog_commit"; then tag="${DIM}[watchdog]${RESET}"
        elif echo "$cmd" | grep -q "local_eval.sh"; then tag="${CYAN}[local_eval]${RESET}"
        elif echo "$cmd" | grep -q "bootstrap_gpu"; then tag="${YELLOW}[bootstrap]${RESET}"
        elif echo "$cmd" | grep -q "ninja\|cc1plus\|nvcc"; then tag="${YELLOW}[compile]${RESET}"
        else tag="[?]"
        fi
        printf "  %s pid=%-6s rss=%-7s etime=%-9s %s\n" "$(echo -e "$tag")" "$pid" "$rss_h" "$etime" "$cmd"
    fi
done < <(ps -e -o user,pid,ppid,%cpu,%mem,rss,vsz,stat,start,etime,command 2>/dev/null | \
    grep -E "sglang.launch_server|quantize_gptqmodel|quantize_gptq_rtn|eval_model.py|watchdog_commit|local_eval.sh|bootstrap_gpu|ninja|cc1plus|nvcc" | \
    grep -v grep)
if [ "$found_any" -eq 0 ]; then
    echo -e "  ${DIM}(no relevant processes — pipeline is idle)${RESET}"
fi
echo

# ---------------------------------------------------------------------------
# 2. GPU usage (nvidia-smi if available)
# ---------------------------------------------------------------------------
hdr "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=, read -r idx name mu mt util pw temp; do
        idx=$(echo "$idx" | xargs); name=$(echo "$name" | xargs)
        mu=$(echo "$mu" | xargs); mt=$(echo "$mt" | xargs)
        util=$(echo "$util" | xargs); pw=$(echo "$pw" | xargs); temp=$(echo "$temp" | xargs)
        # Color util by load
        util_color="$RESET"
        if [ "${util:-0}" -ge 80 ] 2>/dev/null; then util_color="$GREEN"
        elif [ "${util:-0}" -ge 30 ] 2>/dev/null; then util_color="$YELLOW"
        else util_color="$DIM"
        fi
        printf "  GPU %s %-20s mem=%5s/%5s MiB  util=%s%3s%%%s  pwr=%4sW  temp=%2s°C\n" \
            "$idx" "$name" "$mu" "$mt" "$(echo -e $util_color)" "$util" "$RESET" "$pw" "$temp"
    done
else
    echo -e "  ${DIM}(nvidia-smi not available)${RESET}"
fi
echo

# ---------------------------------------------------------------------------
# 3. Latest logs (newest in each category)
# ---------------------------------------------------------------------------
hdr "Latest logs (most recent first)"
latest_log() {
    local dir="$1"
    local pattern="$2"
    if [ -d "$dir" ]; then
        ls -t "$dir"/${pattern} 2>/dev/null | head -1
    fi
}

latest_install=$(latest_log "$PEEK_LOG_DIR" "install_*.log")
latest_exp=$(latest_log "$PEEK_LOG_DIR" "exp_*.log")
latest_quant=$(latest_log "$SCRIPT_LOG_DIR" "local_eval_quant_*.log")
latest_server=$(latest_log "$SCRIPT_LOG_DIR" "local_eval_server_*.log")
latest_eval=$(latest_log "$SCRIPT_LOG_DIR" "local_eval_eval_*.log")
latest_diff=$(latest_log "$PEEK_LOG_DIR" "diff_*.md")
latest_any=$(latest_log "$PEEK_LOG_DIR" "*.log")

[ -n "$latest_install" ] && echo -e "  ${BOLD}install:${RESET} $latest_install ($(stat -c%y "$latest_install" 2>/dev/null | cut -d. -f1))"
[ -n "$latest_quant" ]   && echo -e "  ${BOLD}quant:  ${RESET} $latest_quant ($(stat -c%y "$latest_quant" 2>/dev/null | cut -d. -f1))"
[ -n "$latest_server" ]  && echo -e "  ${BOLD}server: ${RESET} $latest_server ($(stat -c%y "$latest_server" 2>/dev/null | cut -d. -f1))"
[ -n "$latest_eval" ]    && echo -e "  ${BOLD}eval:   ${RESET} $latest_eval ($(stat -c%y "$latest_eval" 2>/dev/null | cut -d. -f1))"
[ -n "$latest_exp" ]     && echo -e "  ${BOLD}exp_*:  ${RESET} $latest_exp ($(stat -c%y "$latest_exp" 2>/dev/null | cut -d. -f1))"
[ -n "$latest_diff" ]    && echo -e "  ${DIM}diff:   ${RESET} $latest_diff"

if [ -z "$latest_install$latest_quant$latest_server$latest_eval$latest_exp" ]; then
    echo -e "  ${DIM}(no log files found under $PEEK_LOG_DIR or $SCRIPT_LOG_DIR)${RESET}"
fi
echo

# ---------------------------------------------------------------------------
# 4. Key signals across the most recent log(s)
# ---------------------------------------------------------------------------
hdr "Key signals (from most recent log of each kind)"
extract_signals() {
    local f="$1"
    [ -f "$f" ] || return 0
    echo -e "  ${DIM}# $(basename "$f")${RESET}"
    # Versions block (compact)
    grep -hE "^\[versions\] (python|torch|transformers|gptqmodel|flash-attn|sglang)" "$f" 2>/dev/null | tail -6 | sed 's/^/    /'
    # Install / pin status
    grep -hE "^\[prepare_env\] (forcing|gptqmodel .* already installed)" "$f" 2>/dev/null | tail -2 | sed 's/^/    /'
    # qzeros-fix summary
    grep -hE "^\[qzeros-fix\] +(qzeros tensors seen:|patched|POST-CHECK|OK|FATAL)" "$f" 2>/dev/null | tail -6 | sed 's/^/    /'
    # DIAGNOSTIC calib + output-dir hints
    grep -hE "^\[prepare_model\] DIAGNOSTIC: calib jsonl (size|row count)|safetensors weight file" "$f" 2>/dev/null | tail -4 | sed 's/^/    /'
    # FATAL / errors near tail
    fatal=$(grep -hE "FATAL|RuntimeError|AssertionError|ImportError|Killed" "$f" 2>/dev/null | tail -3)
    if [ -n "$fatal" ]; then
        echo "$fatal" | sed "s/^/    $(echo -e ${RED})/" | sed "s/$/$(echo -e ${RESET})/"
    fi
}

if [ -n "${latest_install:-}" ];     then extract_signals "$latest_install";     fi
if [ -n "${latest_quant:-}" ];       then extract_signals "$latest_quant";       fi
if [ -n "${latest_server:-}" ];      then extract_signals "$latest_server";      fi
if [ -n "${latest_exp:-}" ];         then extract_signals "$latest_exp";         fi
echo

# ---------------------------------------------------------------------------
# 5. Partial predictions (if eval is running)
# ---------------------------------------------------------------------------
hdr "Partial predictions"
if [ -d "$PEEK_OUTPUTS_DIR" ]; then
    # Top 3 most-recent outputs/ subdirs
    while read -r d; do
        [ -z "$d" ] && continue
        pred="$d/predictions.jsonl"
        if [ -f "$pred" ]; then
            n=$(wc -l < "$pred" 2>/dev/null || echo 0)
            mtime=$(stat -c%y "$pred" 2>/dev/null | cut -d. -f1)
            label="$(basename "$d")"
            # Partial-acc if at least 5 rows + tool is available
            partial=""
            if [ "$n" -ge 5 ] && [ -f "${REPO_ROOT}/tools/compute_corrected_acc.py" ]; then
                line=$(python3 "${REPO_ROOT}/tools/compute_corrected_acc.py" --input "$pred" 2>/dev/null | grep -E "^standard=" | head -1)
                if [ -n "$line" ]; then partial=" ${DIM}[${line}]${RESET}"; fi
            fi
            done_marker=""
            if [ "$n" -ge 150 ]; then done_marker="${GREEN}done${RESET}"
            else done_marker="${YELLOW}partial ($n/150)${RESET}"
            fi
            printf "  %s  %-7s  %s  %s\n" "$(echo -e $done_marker)" "$label" "$mtime" "$(echo -e $partial)"
        fi
    done < <(ls -dt "$PEEK_OUTPUTS_DIR"/*/ 2>/dev/null | head -3)
else
    echo -e "  ${DIM}(no $PEEK_OUTPUTS_DIR)${RESET}"
fi
echo

# ---------------------------------------------------------------------------
# 6. Git state (commit + uncommitted)
# ---------------------------------------------------------------------------
hdr "Git"
if (cd "$REPO_ROOT" && git rev-parse --is-inside-work-tree >/dev/null 2>&1); then
    branch=$(cd "$REPO_ROOT" && git branch --show-current 2>/dev/null)
    head_short=$(cd "$REPO_ROOT" && git log -1 --format='%h %s' 2>/dev/null | head -c 90)
    dirty=$(cd "$REPO_ROOT" && git status --porcelain 2>/dev/null | grep -vE "^\?\?" | wc -l)
    echo -e "  branch: ${BOLD}$branch${RESET}    HEAD: $head_short"
    if [ "$dirty" -gt 0 ]; then
        echo -e "  ${YELLOW}uncommitted: $dirty file(s) modified${RESET}"
    else
        echo -e "  ${GREEN}clean${RESET}"
    fi
else
    echo -e "  ${DIM}(not a git repo)${RESET}"
fi
echo

# ---------------------------------------------------------------------------
# 7. Helpful next-step pointers
# ---------------------------------------------------------------------------
echo -e "${DIM}refresh: bash scripts/peek.sh   ·   auto: bash scripts/peek.sh --watch 10${RESET}"
if [ -n "${latest_quant:-}" ]; then
    echo -e "${DIM}tail:    tail -f $latest_quant${RESET}"
fi
if [ -n "${latest_eval:-}" ]; then
    echo -e "${DIM}tail:    tail -f $latest_eval${RESET}"
fi
