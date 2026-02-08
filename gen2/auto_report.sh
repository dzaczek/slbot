#!/bin/bash
#
# Auto Report Script - Slither.io MatrixBot Training
# Runs analyze_matrix.py every 5 minutes and pushes results to git
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Files to track
FILES_TO_COMMIT=(
    "training_plot.png"
    "training_detailed.png"
    "matrix_stats.csv"
)

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

# Interval in seconds (5 minutes = 300 seconds)
INTERVAL=900

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     🐍 MatrixBot Auto Report - Starting Loop           ║${NC}"
echo -e "${GREEN}║     Interval: ${INTERVAL} seconds (5 minutes)                  ║${NC}"
echo -e "${GREEN}║     Press Ctrl+C to stop                               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Counter for iterations
iteration=0

while true; do
    iteration=$((iteration + 1))

    echo ""
    log "═══════════════════════════════════════════════════════"
    log "📊 Iteration #${iteration} starting..."
    log "═══════════════════════════════════════════════════════"

    # Step 1: Run analyze_matrix.py
    log "Running analyze_matrix.py..."

    if python3 analyze_matrix.py 2>&1; then
        success "Analysis completed"
    else
        error "Analysis failed, but continuing..."
    fi

    # Step 2: Check if there are changes to commit
    log "Checking for changes..."

    cd "$SCRIPT_DIR/.."  # Go to repo root

    # Add the generated files
    changes_found=false
    for file in "${FILES_TO_COMMIT[@]}"; do
        filepath="gen2/$file"
        if [[ -f "$filepath" ]]; then
            git add "$filepath" 2>/dev/null
            if git diff --cached --quiet "$filepath" 2>/dev/null; then
                : # No changes
            else
                changes_found=true
                log "  📄 $file - changed"
            fi
        fi
    done

    # Step 3: Commit and push if there are changes
    if $changes_found; then
        log "Committing changes..."

        # Create commit message with timestamp
        commit_msg="auto: update training reports $(date '+%Y-%m-%d %H:%M')"

        if git commit -m "$commit_msg" 2>&1; then
            success "Committed: $commit_msg"

            # Push to origin
            log "Pushing to origin/main..."
            if git push origin main 2>&1; then
                success "Pushed successfully!"
            else
                error "Push failed. Will retry next iteration."
            fi
        else
            error "Commit failed"
        fi
    else
        log "No changes to commit"
    fi

    cd "$SCRIPT_DIR"  # Return to gen2 directory

    # Step 4: Wait for next iteration
    log "Next update in ${INTERVAL} seconds ($(date -d "+${INTERVAL} seconds" '+%H:%M:%S' 2>/dev/null || date -v+${INTERVAL}S '+%H:%M:%S'))"
    log "═══════════════════════════════════════════════════════"

    sleep $INTERVAL
done
