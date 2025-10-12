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
        echo "Stopping all File Organizer processes..."
        STOPPED=0
        
        # First, try to stop daemon (using PID file)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Stopping daemon (PID: $PID)..."
                kill "$PID"
                sleep 1
                # Force kill if still running
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID" 2>/dev/null
                fi
                rm "$PID_FILE"
                echo "  ✓ Daemon stopped"
                STOPPED=1
            else
                rm "$PID_FILE"
            fi
        fi
        
        # Find and stop ALL file_organizer.py processes
        # This catches --scan-once, test mode, stuck rsync, or orphaned processes
        PIDS=$(pgrep -f "python.*file_organizer.py" 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            for PID in $PIDS; do
                # Check if process is still running and is file_organizer
                if ps -p "$PID" > /dev/null 2>&1; then
                    CMDLINE=$(ps -p "$PID" -o command= 2>/dev/null || true)
                    if echo "$CMDLINE" | grep -q "file_organizer.py"; then
                        echo "Stopping file_organizer (PID: $PID)..."
                        kill "$PID" 2>/dev/null
                        sleep 1
                        # Force kill if still running
                        if ps -p "$PID" > /dev/null 2>&1; then
                            kill -9 "$PID" 2>/dev/null
                            echo "  ✓ Force stopped (was stuck)"
                        else
                            echo "  ✓ Stopped gracefully"
                        fi
                        STOPPED=1
                    fi
                fi
            done
        fi
        
        # Also kill any stuck rsync child processes
        RSYNC_PIDS=$(pgrep -f "rsync.*GoogleDrive|rsync.*ProtonDrive|rsync.*PASSPORT4" 2>/dev/null || true)
        if [ -n "$RSYNC_PIDS" ]; then
            for PID in $RSYNC_PIDS; do
                if ps -p "$PID" > /dev/null 2>&1; then
                    echo "Stopping stuck rsync (PID: $PID)..."
                    kill -9 "$PID" 2>/dev/null
                    echo "  ✓ Killed stuck rsync"
                    STOPPED=1
                fi
            done
        fi
        
        if [ $STOPPED -eq 0 ]; then
            echo "No File Organizer processes found"
        else
            echo ""
            echo "All File Organizer processes stopped"
            echo "Progress saved - restart with: python3 file_organizer.py -R --scan-once"
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
