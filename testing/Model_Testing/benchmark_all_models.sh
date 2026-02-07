#!/usr/bin/env bash
set -euo pipefail

# ---------- Colors ----------
RESET=$'\033[0m'
BOLD=$'\033[1m'
DIM=$'\033[2m'

GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
CYAN=$'\033[36m'

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC_DIR="$PROJECT_ROOT/src/Email_Classification/Models"
SAMPLE_FILE="$SCRIPT_DIR/sample_email.txt"

# ---------- Column widths ----------
COL1=30
COL2=12
COL3=90

print_line() {
    printf "+"
    printf "%0.s-" $(seq 1 $((COL1+2)))
    printf "+"
    printf "%0.s-" $(seq 1 $((COL2+2)))
    printf "+"
    printf "%0.s-" $(seq 1 $((COL3+2)))
    printf "+\n"
}

truncate() {
    local text="$1"
    local max="$2"

    if [ "${#text}" -gt "$max" ]; then
        printf "%s" "${text:0:$((max-3))}..."
    else
        printf "%-${max}s" "$text"
    fi
}

echo
printf "${BOLD}${CYAN}=== Benchmark Summary ===${RESET}\n\n"

print_line
printf "| %-*s | %-*s | %-*s |\n" \
    $COL1 "File" \
    $COL2 "Time (s)" \
    $COL3 "Output"
print_line

# ---------- Total timer start ----------
total_start=$(date +%s.%N)

for file in "$SRC_DIR"/*.py; do
    filename=$(basename "$file")

    start=$(date +%s.%N)
    full_output="$("$PROJECT_ROOT/.venv/bin/python" "$file" "$SAMPLE_FILE" 2>&1)"
    status=$?
    end=$(date +%s.%N)

    elapsed=$(echo "$end - $start" | bc)

    # ---------- Robust tag extraction ----------
    tag_line=$(echo "$full_output" | grep -m1 'tag:' || true)

    if [ -n "$tag_line" ]; then
        # Remove everything before "tag:"
        cleaned=$(echo "$tag_line" | sed -E 's/.*tag:\s*//')

        # Remove brackets if present
        cleaned=$(echo "$cleaned" | sed -E 's/^\[//; s/\]$//')

        # Trim whitespace
        output_text=$(echo "$cleaned" | xargs)
    else
        output_text="No Tag Found"
    fi

    # ---------- Format first, then color ----------
    file_plain=$(truncate "$filename" "$COL1")
    time_plain=$(printf "%-${COL2}.6f" "$elapsed")
    output_plain=$(truncate "$output_text" "$COL3")

    if [ $status -eq 0 ]; then
        time_colored="${GREEN}${time_plain}${RESET}"
        output_colored="${YELLOW}${output_plain}${RESET}"
    else
        time_colored="${RED}${time_plain}${RESET}"
        output_colored="${RED}${output_plain}${RESET}"
    fi

    printf "| %s | %s | %s |\n" \
        "$file_plain" \
        "$time_colored" \
        "$output_colored"

    print_line
done

# ---------- Total timer end ----------
total_end=$(date +%s.%N)
total_elapsed=$(echo "$total_end - $total_start" | bc)

total_time_plain=$(printf "%-${COL2}.6f" "$total_elapsed")
total_label_plain=$(truncate "TOTAL EXECUTION TIME" "$COL1")
empty_output=$(printf "%-${COL3}s" "")

printf "| %s | ${BOLD}${CYAN}%s${RESET} | %s |\n" \
    "$total_label_plain" \
    "$total_time_plain" \
    "$empty_output"

print_line
echo
