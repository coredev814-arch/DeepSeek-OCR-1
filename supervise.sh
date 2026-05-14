#!/bin/bash
# Auto-restart supervisor for DeepSeek-OCR API.
# Restarts the service when:
#   1. Python process exits (crash, OOM, etc.)
#   2. /health returns "status":"dead" (vLLM engine background loop died)
#   3. /health is unreachable for >2 consecutive minutes (deadlock)
#
# Usage: ./supervise.sh
# Logs:  $LOG_FILE (default: /workspace/logs/api.log)
# Quit:  Ctrl-C / SIGTERM — propagates to the child and exits.

set -u

PORT="${PORT:-8000}"
HEALTH_URL="http://localhost:${PORT}/health"
LOG_FILE="${LOG_FILE:-/workspace/logs/api.log}"
START_SCRIPT="$(dirname "$0")/start.sh"
POLL_INTERVAL=30          # seconds between /health checks
UNREACHABLE_LIMIT=4       # consecutive failed checks before restart (4*30s = 2min)
STARTUP_GRACE=120         # seconds after launch before health checks count
RESTART_BACKOFF=5         # seconds to sleep between restarts

mkdir -p "$(dirname "$LOG_FILE")"

CHILD_PID=""
SHUTDOWN=0

log() { printf '[supervise %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

shutdown() {
    SHUTDOWN=1
    log "shutdown signal received"
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        log "stopping child PID $CHILD_PID"
        kill -TERM "$CHILD_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$CHILD_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$CHILD_PID" 2>/dev/null
    fi
    exit 0
}
trap shutdown INT TERM

start_child() {
    log "launching ${START_SCRIPT}"
    "$START_SCRIPT" >> "$LOG_FILE" 2>&1 &
    CHILD_PID=$!
    log "child PID=$CHILD_PID"
}

kill_child() {
    [[ -z "$CHILD_PID" ]] && return
    if kill -0 "$CHILD_PID" 2>/dev/null; then
        log "killing child PID $CHILD_PID"
        # also nuke any python3 api_service descendants vLLM may have spawned
        pkill -TERM -P "$CHILD_PID" 2>/dev/null
        kill -TERM "$CHILD_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$CHILD_PID" 2>/dev/null || break
            sleep 1
        done
        pkill -9 -P "$CHILD_PID" 2>/dev/null
        kill -9 "$CHILD_PID" 2>/dev/null
        # belt-and-braces: any leftover python3 api_service process
        pkill -9 -f "python3 api_service.py" 2>/dev/null
    fi
    CHILD_PID=""
}

while [[ $SHUTDOWN -eq 0 ]]; do
    start_child
    started_at=$(date +%s)
    unreachable_streak=0

    while [[ $SHUTDOWN -eq 0 ]]; do
        sleep "$POLL_INTERVAL"

        # 1. Process exited?
        if ! kill -0 "$CHILD_PID" 2>/dev/null; then
            log "child PID $CHILD_PID exited; will restart"
            CHILD_PID=""
            break
        fi

        # 2. Still inside startup grace period — skip health logic
        now=$(date +%s)
        if (( now - started_at < STARTUP_GRACE )); then
            continue
        fi

        # 3. Probe /health
        body=$(curl -sS --max-time 5 "$HEALTH_URL" 2>/dev/null || true)
        if [[ -z "$body" ]]; then
            unreachable_streak=$((unreachable_streak + 1))
            log "health unreachable (${unreachable_streak}/${UNREACHABLE_LIMIT})"
            if (( unreachable_streak >= UNREACHABLE_LIMIT )); then
                log "health unreachable for $((UNREACHABLE_LIMIT * POLL_INTERVAL))s — restarting"
                kill_child
                break
            fi
            continue
        fi
        unreachable_streak=0

        if grep -q '"status":"dead"' <<<"$body"; then
            log "health says engine dead — restarting"
            kill_child
            break
        fi
    done

    [[ $SHUTDOWN -eq 0 ]] && sleep "$RESTART_BACKOFF"
done
