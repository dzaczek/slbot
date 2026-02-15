#!/bin/bash
#
# Auto Report Script - SlitherBot Gen2 Training
# Runs training_progress_analyzer.py every 10 minutes and pushes to GitHub
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
cd "$SCRIPT_DIR"

# Disable git pager (no 'q' to confirm)
export GIT_PAGER=cat

# Activate venv
source /Users/dzaczek/slbot/venv/bin/activate

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }

INTERVAL=3600

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SlitherBot Auto Report (${INTERVAL}s)${NC}"
echo -e "${GREEN}  Repo: ${REPO_ROOT}${NC}"
echo -e "${GREEN}  Ctrl+C to stop${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

iteration=0

while true; do
    iteration=$((iteration + 1))
    log "--- Iteration #${iteration} ---"

    # Step 1: Run analyzer
    log "Running analyzer..."
    python3 "$SCRIPT_DIR/training_progress_analyzer.py" --latest 2>&1 | tail -3

    # Step 2: Stage files using git -C (never cd away)
    FILES=(
        training_stats.csv
        progress_report.md
        logs/train.log
        logs/app.log
    )

    added=0
    for f in "${FILES[@]}"; do
        full="$SCRIPT_DIR/$f"
        if [[ -f "$full" ]]; then
            git -C "$REPO_ROOT" add -f "gen2/$f" 2>&1
            added=$((added + 1))
        fi
    done

    # Auto-add all chart PNGs (catches new charts automatically)
    for png in "$SCRIPT_DIR"/chart_*.png; do
        [[ -f "$png" ]] || continue
        fname="$(basename "$png")"
        git -C "$REPO_ROOT" add -f "gen2/$fname" 2>&1
        added=$((added + 1))
    done
    log "Staged $added files"

    # Step 3: Check & commit & push
    if git -C "$REPO_ROOT" diff --cached --quiet; then
        log "No changes to commit"
    else
        git -C "$REPO_ROOT" diff --cached --stat
        git -C "$REPO_ROOT" commit -m "stats" 2>&1 | tail -1
        success "Committed"
        if git -C "$REPO_ROOT" push origin main 2>&1; then
            success "Pushed to GitHub"
        else
            error "Push failed"
        fi
    fi

    next_time=$(date -v+${INTERVAL}S '+%H:%M:%S' 2>/dev/null || echo "~10min")
    log "Next: ${next_time}"
    log "---"
    echo ""

    sleep $INTERVAL
done
