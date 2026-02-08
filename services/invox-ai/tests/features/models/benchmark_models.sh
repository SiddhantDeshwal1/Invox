#!/usr/bin/env bash
set -euo pipefail

# ================= CONFIG =================

DATASET_DIR="/run/media/man37/Alpha/project/Invox/services/invox-ai/tests/features/models/email_dataset"
ANSWERS_FILE="$DATASET_DIR/answers.txt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/services/invox-ai/src/invox/models/experimental"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

# ================= COLORS =================

RESET=$'\033[0m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
GREEN=$'\033[32m'
RED=$'\033[31m'
CYAN=$'\033[36m'
YELLOW=$'\033[33m'
BLUE=$'\033[34m'

# ================= TERMINAL SAFETY =================

cleanup_terminal() {
    stty echo 2>/dev/null || true
    tput cnorm 2>/dev/null || true
}
trap cleanup_terminal EXIT

# ================= SPINNER =================

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

# ================= TABLE WIDTHS =================

COL1=34
COL2=20
COL3=10
COL4=7

print_line() {
    printf "+"
    printf "%0.s-" $(seq 1 $((COL1+2)))
    printf "+"
    printf "%0.s-" $(seq 1 $((COL2+2)))
    printf "+"
    printf "%0.s-" $(seq 1 $((COL3+2)))
    printf "+"
    printf "%0.s-" $(seq 1 $((COL4+2)))
    printf "+\n"
}

# ================= ARG PARSE =================

MODE="$1"
LIMIT="$2"

# ================= MODEL COLLECTION =================

mapfile -t ALL_MODELS < <(
    find "$MODEL_DIR" -maxdepth 1 -type f -name "*.py" ! -name "__init__.py" | sort
)

if [ "$MODE" = "--allmodel" ]; then
    MODELS=("${ALL_MODELS[@]}")
elif [ "$MODE" = "--custom" ]; then
    echo
    printf "${BOLD}${CYAN}Available Models${RESET}\n"
    echo "--------------------------------"
    for i in "${!ALL_MODELS[@]}"; do
        printf "%2d) %s\n" $((i+1)) "$(basename "${ALL_MODELS[$i]}")"
    done
    echo
    read -p "Select models (e.g. 1,2): " selection
    IFS=',' read -ra INDICES <<< "$selection"
    MODELS=()
    for index in "${INDICES[@]}"; do
        MODELS+=("${ALL_MODELS[$((index-1))]}")
    done
fi

# ================= EMAIL COLLECTION =================

mapfile -t EMAILS < <(find "$DATASET_DIR" -type f ! -name "answers.txt" | sort)
mapfile -t ANSWERS < "$ANSWERS_FILE"

TOTAL_EMAILS=${#EMAILS[@]}

if [ "$LIMIT" = "-all" ]; then
    SELECTED=($(seq 0 $((TOTAL_EMAILS-1))))
else
    COUNT="${LIMIT#-}"
    [ "$COUNT" -gt "$TOTAL_EMAILS" ] && COUNT=$TOTAL_EMAILS
    SELECTED=($(shuf -i 0-$((TOTAL_EMAILS-1)) -n "$COUNT"))
fi

# ================= METRICS =================

declare -A PASS
declare -A FAIL
declare -A TOTAL_TIME

# ================= HEADER =================

echo
printf "${BOLD}${BLUE}============= EMAIL MODEL BENCHMARK =============${RESET}\n"
printf "${DIM}Mode:${RESET} %s\n" "$MODE"
printf "${DIM}Emails Selected:${RESET} %d\n\n" "${#SELECTED[@]}"

# ================= BENCHMARK =================

TOTAL_SELECTED=${#SELECTED[@]}
current_email=0

for idx in "${SELECTED[@]}"; do

    current_email=$((current_email + 1))

    email="${EMAILS[$idx]}"
    actual="${ANSWERS[$idx]}"
    filename=$(basename "$email")

    BOX_WIDTH=60
    TOTAL_WIDTH=$((BOX_WIDTH + 2))

    printf "\n${BLUE}"
    printf "┌"
    printf "%0.s─" $(seq 1 $TOTAL_WIDTH)
    printf "┐\n"
    printf "│ %-*s │\n" $BOX_WIDTH "Email $current_email / $TOTAL_SELECTED"
    printf "│ %-*s │\n" $BOX_WIDTH "File: $filename"
    printf "│ %-*s │\n" $BOX_WIDTH "Actual Label: $actual"
    printf "└"
    printf "%0.s─" $(seq 1 $TOTAL_WIDTH)
    printf "┘${RESET}\n\n"

    print_line
    printf "| %-*s | %-*s | %-*s | %-*s |\n" \
        $COL1 "Model" \
        $COL2 "Prediction" \
        $COL3 "Time(s)" \
        $COL4 "Match"
    print_line

    for model in "${MODELS[@]}"; do

        model_name=$(basename "$model")

        start=$(date +%s.%N)
        "$PYTHON_BIN" "$model" "$email" > /tmp/model_out.txt 2>&1 &
        pid=$!

        spinner_with_label "$pid" "$model_name"
        wait $pid
        end=$(date +%s.%N)

        elapsed=$(echo "$end - $start" | bc)
        output=$(cat /tmp/model_out.txt)

        tag_line=$(echo "$output" | grep -m1 'tag:' || true)

        if [ -n "$tag_line" ]; then
            prediction=$(echo "$tag_line" | sed -E 's/.*tag:\s*//' | sed -E 's/^\[//; s/\]$//' | xargs)
        else
            prediction="None"
        fi

        if [ "$prediction" = "$actual" ]; then
            PASS["$model_name"]=$(( ${PASS["$model_name"]:-0} + 1 ))
            match="TRUE"
            match_color="${GREEN}"
        else
            FAIL["$model_name"]=$(( ${FAIL["$model_name"]:-0} + 1 ))
            match="FALSE"
            match_color="${RED}"
        fi

        TOTAL_TIME["$model_name"]=$(echo "${TOTAL_TIME["$model_name"]:-0} + $elapsed" | bc)

        printf "| %-*s | %-*s | %-*.3f | %-*s |\n" \
            $COL1 "$model_name" \
            $COL2 "$prediction" \
            $COL3 "$elapsed" \
            $COL4 "$match"

        print_line
    done
done

printf "\n${BOLD}${CYAN}================ FINAL METRICS =================${RESET}\n\n"
