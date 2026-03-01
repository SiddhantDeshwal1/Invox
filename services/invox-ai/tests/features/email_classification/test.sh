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

DATASET_DIR="$SCRIPT_DIR/email_dataset"
ANSWERS_FILE="$DATASET_DIR/answer.txt"

SERVICE_PATH="$PROJECT_ROOT/services/invox-ai/src/invox/features/email_classification/service.py"
export PYTHONPATH="$PROJECT_ROOT/services/invox-ai/src"

# Use the globally active environment (Conda in Lightning Studio)
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

# ===================== VALIDATE DATASET =====================
[[ -d "$DATASET_DIR" ]] || error_exit "Dataset directory missing: $DATASET_DIR"
[[ -f "$ANSWERS_FILE" ]] || error_exit "answers.txt missing in $DATASET_DIR"

mapfile -t ANSWERS < "$ANSWERS_FILE"
if [ ${#ANSWERS[@]} -eq 0 ]; then
    error_exit "answers.txt is empty"
fi

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
TOTAL_TIME_SEC=0
TOTAL_RUNS=0

# ===================== HEADER =====================
echo
echo -e "${BOLD}${BLUE}========= EMAIL CLASSIFICATION BENCHMARK =========${RESET}"
echo -e "${YELLOW}Loading AI model into VRAM... (This takes a few seconds)${RESET}"
echo
printf "+--------------+------------------------+-----------+------------------------+--------+\n"
printf "| %-12s | %-22s | %-9s | %-22s | %-6s |\n" \
    "Email" "Prediction" "Time" "Actual" "Match"
printf "+--------------+------------------------+-----------+------------------------+--------+\n"

# ===================== MAIN BATCH STREAM =====================

# We run Python ONCE with all files. 
# It streams data out. The while loop catches it and draws the table in real-time.
while IFS='|' read -r prefix filename pred_cat pred_sub elapsed_sec; do
    
    # Ignore any standard logs, only process lines starting with "RESULT|"
    if [[ "$prefix" != "RESULT" ]]; then
        continue
    fi

    id="${filename%.txt}"
    index=$((10#$id - 1))
    
    if [ "$index" -ge "${#ANSWERS[@]}" ]; then
         error_exit "Missing answer for $filename (Index $index exceeds answers.txt length)"
    fi

    actual_line="${ANSWERS[$index]}"
    
    if [[ "$actual_line" == *","* ]]; then
        actual_cat=$(echo "$actual_line" | cut -d',' -f1 | xargs)
        actual_sub=$(echo "$actual_line" | cut -d',' -f2 | xargs)
    else
        actual_cat=$(echo "$actual_line" | awk -F ' - ' '{print $1}' | xargs)
        actual_sub=$(echo "$actual_line" | awk -F ' - ' '{print $2}' | xargs)
    fi

    prediction="${pred_cat},${pred_sub}"
    actual="${actual_cat},${actual_sub}"

    # Score calculation
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
    TOTAL_TIME_SEC=$(awk -v total="$TOTAL_TIME_SEC" -v current="$elapsed_sec" 'BEGIN {print total + current}')

    printf "| %-12s | %-22s | %-9s | %-22s | ${color}%-6s${RESET} |\n" \
        "$filename" "$prediction" "${elapsed_sec}s" "$actual" "$match"

    printf "+--------------+------------------------+-----------+------------------------+--------+\n"

done < <("$PYTHON_BIN" "$SERVICE_PATH" "${EMAILS[@]}" 2>/dev/null) 
# Note: 2>/dev/null hides the massive Hugging Face warnings from destroying the table visually.

if [ "$TOTAL_RUNS" -eq 0 ]; then
    error_exit "No emails processed. Check Python script."
fi

avg_sec=$(awk -v total="$TOTAL_TIME_SEC" -v runs="$TOTAL_RUNS" 'BEGIN {printf "%.3f", total / runs}')

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
