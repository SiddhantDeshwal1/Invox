#!/usr/bin/env bash
set -euo pipefail

# ===================== ERROR HANDLER =====================

error_exit() {
    echo -e "\n\033[31mERROR:\033[0m $1"
    cleanup_terminal
    exit 1
}

# ===================== CONFIG =====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Point to the local email_dataset next to the script
DATASET_DIR="$SCRIPT_DIR/email_dataset"
ANSWERS_FILE="$DATASET_DIR/answer.txt"

SERVICE_PATH="$PROJECT_ROOT/services/invox-ai/src/invox/features/email_classification/service.py"
export PYTHONPATH="$PROJECT_ROOT/services/invox-ai/src"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

# ===================== COLORS =====================

RESET=$'\033[0m'
BOLD=$'\033[1m'
BLUE=$'\033[34m'
GREEN=$'\033[32m'
RED=$'\033[31m'
YELLOW=$'\033[33m'
CYAN=$'\033[36m'

# ===================== TERMINAL SAFETY =====================

cleanup_terminal() {
    stty echo 2>/dev/null || true
    tput cnorm 2>/dev/null || true
}
trap cleanup_terminal EXIT

spinner_with_label() {
    local pid=$1
    local label="$2"
    local spin='|/-\'

    stty -echo -icanon 2>/dev/null || true
    tput civis 2>/dev/null || true

    while kill -0 "$pid" 2>/dev/null; do
        for i in {0..3}; do
            printf "\r${CYAN}Running %-30s ${spin:$i:1}${RESET}" "$label"
            sleep 0.08
        done
    done

    printf "\r%*s\r" 80 ""
    cleanup_terminal
}

# ===================== VALIDATE DATASET =====================

[[ -d "$DATASET_DIR" ]] || error_exit "Dataset directory missing: $DATASET_DIR"
[[ -f "$ANSWERS_FILE" ]] || error_exit "answers.txt missing in $DATASET_DIR"

mapfile -t ANSWERS < "$ANSWERS_FILE"
if [ ${#ANSWERS[@]} -eq 0 ]; then
    error_exit "answers.txt is empty"
fi

# Collect only numbered txt files (supports both 01.txt and 001.txt)
ALL_EMAILS=()
while IFS= read -r file; do
    base=$(basename "$file")
    if [[ "$base" =~ ^[0-9]{2,3}\.txt$ ]]; then
        ALL_EMAILS+=("$file")
    fi
done < <(find "$DATASET_DIR" -maxdepth 1 -type f -name "*.txt" | sort)

if [ ${#ALL_EMAILS[@]} -eq 0 ]; then
    error_exit "No numbered .txt files found"
fi

# ===================== RANDOM SELECTION =====================

LIMIT="${1:-"-all"}"

if [[ "$LIMIT" != "-all" ]]; then
    COUNT="${LIMIT#-}"
    [[ "$COUNT" =~ ^[0-9]+$ ]] || error_exit "Use -N or -all"
    if [ "$COUNT" -eq 0 ]; then
        error_exit "Count must be > 0"
    fi

    if [ "$COUNT" -gt ${#ALL_EMAILS[@]} ]; then
        COUNT=${#ALL_EMAILS[@]}
    fi

    mapfile -t EMAILS < <(printf "%s\n" "${ALL_EMAILS[@]}" | shuf | head -n "$COUNT")
else
    EMAILS=("${ALL_EMAILS[@]}")
fi

# ===================== METRICS =====================

FULL_PASS=0
PARTIAL_PASS=0
FULL_FAIL=0
TOTAL_TIME_NS=0
TOTAL_RUNS=0

# ===================== HEADER =====================

echo
echo -e "${BOLD}${BLUE}========= EMAIL CLASSIFICATION BENCHMARK =========${RESET}"
echo
printf "+--------------+------------------------+-----------+------------------------+--------+\n"
printf "| %-12s | %-22s | %-9s | %-22s | %-6s |\n" \
    "Email" "Prediction" "Time" "Actual" "Match"
printf "+--------------+------------------------+-----------+------------------------+--------+\n"

# ===================== MAIN LOOP =====================

for email in "${EMAILS[@]}"; do

    filename=$(basename "$email")
    id="${filename%.txt}"

    # Base 10 conversion to prevent octal errors on 008/009
    index=$((10#$id - 1))
    
    if [ "$index" -ge "${#ANSWERS[@]}" ]; then
         error_exit "Missing answer for $filename (Index $index exceeds answers.txt length)"
    fi

    actual_line="${ANSWERS[$index]}"
    
    # Auto-detect comma vs hyphen formatted answers
    if [[ "$actual_line" == *","* ]]; then
        actual_cat=$(echo "$actual_line" | cut -d',' -f1 | xargs)
        actual_sub=$(echo "$actual_line" | cut -d',' -f2 | xargs)
    else
        actual_cat=$(echo "$actual_line" | awk -F ' - ' '{print $1}' | xargs)
        actual_sub=$(echo "$actual_line" | awk -F ' - ' '{print $2}' | xargs)
    fi

    tmpfile=$(mktemp)
    start_ns=$(date +%s%N)

    "$PYTHON_BIN" "$SERVICE_PATH" "$email" > "$tmpfile" 2>&1 &
    pid=$!
    spinner_with_label "$pid" "$filename"
    wait $pid || true 

    end_ns=$(date +%s%N)
    elapsed_ns=$((end_ns - start_ns))
    TOTAL_TIME_NS=$((TOTAL_TIME_NS + elapsed_ns))

    elapsed_sec=$(awk -v ns="$elapsed_ns" 'BEGIN {printf "%.2f", ns / 1000000000}')
    output=$(cat "$tmpfile")
    rm -f "$tmpfile"

    pred_cat=$(echo "$output" | grep -i '^cat:' | sed -E 's/^cat:\s*//' | xargs || true)
    pred_sub=$(echo "$output" | grep -i '^subcat:' | sed -E 's/^subcat:\s*//' | xargs || true)

    if [[ -z "$pred_cat" ]]; then
         pred_cat="Error"
         pred_sub="ParsingFailed"
    fi

    prediction="${pred_cat},${pred_sub}"
    actual="${actual_cat},${actual_sub}"

    # Safe Arithmetic (No ((VAR++)) so set -e doesn't kill the script)
    if [[ "$pred_cat" == "$actual_cat" && "$pred_sub" == "$actual_sub" ]]; then
        match=100
        color=$GREEN
        FULL_PASS=$((FULL_PASS + 1))
    elif [[ "$pred_cat" == "$actual_cat" ]]; then
        match=50
        color=$YELLOW
        PARTIAL_PASS=$((PARTIAL_PASS + 1))
    else
        match=0
        color=$RED
        FULL_FAIL=$((FULL_FAIL + 1))
    fi

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    printf "| %-12s | %-22s | %-9s | %-22s | ${color}%-6s${RESET} |\n" \
        "$filename" "$prediction" "${elapsed_sec}s" "$actual" "$match"

    printf "+--------------+------------------------+-----------+------------------------+--------+\n"

done

if [ "$TOTAL_RUNS" -eq 0 ]; then
    error_exit "No emails processed"
fi

avg_sec=$(awk -v ns="$TOTAL_TIME_NS" -v runs="$TOTAL_RUNS" 'BEGIN {printf "%.3f", ns / runs / 1000000000}')

# ===================== FINAL METRICS =====================

echo
echo -e "${BOLD}${CYAN}============= FINAL METRICS =============${RESET}"
echo
printf "Full Pass   : ${GREEN}%d${RESET}\n" "$FULL_PASS"
printf "Partial Pass: ${YELLOW}%d${RESET}\n" "$PARTIAL_PASS"
printf "Full Fail   : ${RED}%d${RESET}\n" "$FULL_FAIL"
printf "Total Runs  : %d\n" "$TOTAL_RUNS"
printf "Avg Time    : %s sec\n" "$avg_sec"
echo
