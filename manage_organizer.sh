#!/bin/bash
# Management script for File Organizer

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_NAME="file_organizer.py"
LOG_FILE="$HOME/.file_organizer.log"
PID_FILE="/tmp/file_organizer.pid"

cd "$SCRIPT_DIR" || exit 1

case "$1" in
    start)
        echo "Starting File Organizer in PRODUCTION MODE..."
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "File Organizer is already running (PID: $PID)"
                exit 1
            fi
        fi
        nohup python3 "$SCRIPT_NAME" --REAL > /dev/null 2>&1 &
        echo $! > "$PID_FILE"
        echo "File Organizer started in PRODUCTION MODE (PID: $(cat $PID_FILE))"
        echo "Log file: $LOG_FILE"
        ;;
    
    stop)
        echo "Stopping File Organizer..."
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                kill "$PID"
                rm "$PID_FILE"
                echo "File Organizer stopped"
            else
                echo "File Organizer is not running"
                rm "$PID_FILE"
            fi
        else
            echo "File Organizer is not running"
        fi
        ;;
    
    restart)
        "$0" stop
        sleep 2
        "$0" start
        ;;
    
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "File Organizer is running (PID: $PID)"
                echo "Log file: $LOG_FILE"
            else
                echo "File Organizer is not running (stale PID file)"
                rm "$PID_FILE"
            fi
        else
            echo "File Organizer is not running"
        fi
        ;;
    
    log)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "Log file not found: $LOG_FILE"
        fi
        ;;
    
    test)
        echo "Running single scan in TEST MODE..."
        python3 "$SCRIPT_NAME" --scan-once
        ;;
    
    test-real)
        echo "Running single scan in PRODUCTION MODE..."
        python3 "$SCRIPT_NAME" --REAL --scan-once
        ;;
    
    sync)
        echo "Running folder synchronization only (PRODUCTION MODE)..."
        python3 "$SCRIPT_NAME" --REAL --sync-only
        ;;
    
    dedupe)
        echo "Running duplicate detection and removal (PRODUCTION MODE)..."
        python3 "$SCRIPT_NAME" --REAL --dedupe-only
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|log|test|test-real|sync|dedupe}"
        echo ""
        echo "Background Daemon Commands:"
        echo "  start     - Start organizer as background daemon (PRODUCTION MODE)"
        echo "  stop      - Stop the background daemon"
        echo "  restart   - Restart the background daemon"
        echo "  status    - Check if daemon is running"
        echo "  log       - Tail the log file (for background daemon only)"
        echo ""
        echo "Interactive Commands (see output in terminal):"
        echo "  test      - Run single scan (TEST MODE) - interactive"
        echo "  test-real - Run single scan (PRODUCTION MODE) - interactive"
        echo "  sync      - Synchronize folders only (PRODUCTION MODE) - interactive"
        echo "  dedupe    - Remove duplicates only (PRODUCTION MODE) - interactive"
        echo ""
        echo "Note: Interactive commands show output directly. Background daemon"
        echo "      logs to ~/.file_organizer.log (use 'log' command to monitor)."
        exit 1
        ;;
esac

exit 0
