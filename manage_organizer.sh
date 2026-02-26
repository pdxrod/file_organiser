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
        
        # Validate config file before starting (run validation in foreground to show errors)
        echo "Validating configuration file..."
        VALIDATION_OUTPUT=$(python "$SCRIPT_NAME" --REAL --validate-config 2>&1)
        VALIDATION_EXIT=$?
        
        if [ $VALIDATION_EXIT -ne 0 ]; then
            echo ""
            echo "$VALIDATION_OUTPUT"
            echo ""
            exit 1
        fi
        
        echo "✓ Configuration file is valid"
        
        # Config is valid, now start in background
        # Use --no-daemon so Python runs as single process (PID stays valid); logging goes to file
        nohup python "$SCRIPT_NAME" --REAL --no-daemon > /dev/null 2>&1 &
        PID=$!
        echo $PID > "$PID_FILE"
        
        # Give it a moment to start, then check if it's still running
        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "File Organizer started in PRODUCTION MODE (PID: $PID)"
            echo "Log file: $LOG_FILE"
        else
            echo "ERROR: File Organizer failed to start!"
            echo "Check the log file for details: $LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi
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
        # Use ps + grep instead of pgrep for better pattern matching
        PIDS=$(ps auxw | grep -i "file_organizer.py" | grep -v grep | awk '{print $2}' || true)
        if [ -n "$PIDS" ]; then
            for PID in $PIDS; do
                # Double-check it's really a file_organizer process
                if ps -p "$PID" > /dev/null 2>&1; then
                    CMDLINE=$(ps -p "$PID" -o command= 2>/dev/null || true)
                    if echo "$CMDLINE" | grep -q "file_organizer.py"; then
                        echo "Stopping file_organizer (PID: $PID)..."
                        # Get process start time to show how long it's been running
                        ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | xargs || echo "unknown")
                        echo "  Running for: $ELAPSED"
                        
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
        
        # Also kill any stuck rsync child processes related to our sync operations
        RSYNC_PIDS=$(ps auxw | grep "rsync" | grep -E "GoogleDrive|ProtonDrive|PASSPORT4" | grep -v grep | awk '{print $2}' || true)
        if [ -n "$RSYNC_PIDS" ]; then
            for PID in $RSYNC_PIDS; do
                if ps -p "$PID" > /dev/null 2>&1; then
                    ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | xargs || echo "unknown")
                    echo "Stopping stuck rsync (PID: $PID, running: $ELAPSED)..."
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
            echo "Progress saved - restart with: python file_organizer.py -R --scan-once"
        fi
        ;;
    
    restart)
        "$0" stop
        sleep 2
        "$0" start
        ;;
    
    status)
        RUNNING=0
        
        # Check daemon (PID file)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Daemon is running (PID: $PID)"
                echo "  Log file: $LOG_FILE"
                RUNNING=1
            else
                echo "Daemon PID file exists but process not running (stale)"
                rm "$PID_FILE"
            fi
        fi
        
        # Check for any other file_organizer processes
        PIDS=$(ps auxw | grep -i "file_organizer.py" | grep -v grep | awk '{print $2}' || true)
        if [ -n "$PIDS" ]; then
            for PID in $PIDS; do
                if ps -p "$PID" > /dev/null 2>&1; then
                    ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | xargs || echo "unknown")
                    CMDLINE=$(ps -p "$PID" -o command= 2>/dev/null | cut -c 1-80 || echo "")
                    echo "Process running (PID: $PID, elapsed: $ELAPSED)"
                    echo "  Command: $CMDLINE"
                    RUNNING=1
                fi
            done
        fi
        
        if [ $RUNNING -eq 0 ]; then
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
        python "$SCRIPT_NAME" --scan-once
        ;;
    
    test-real)
        echo "Running single scan in PRODUCTION MODE..."
        python "$SCRIPT_NAME" --REAL --scan-once
        ;;
    
    sync)
        echo "Running folder synchronization only (PRODUCTION MODE)..."
        python "$SCRIPT_NAME" --REAL --sync-only
        ;;
    
    dedupe)
        echo "Running duplicate detection and removal (PRODUCTION MODE)..."
        python "$SCRIPT_NAME" --REAL --dedupe-only
        ;;

    cleanup)
        echo "Cleaning up broken and excluded symlinks in ~/organized..."
        python "$SCRIPT_NAME" --REAL --cleanup
        ;;

    gui)
        echo "Starting File Organizer Desktop App..."
        # Try to bring window to front on Mac
        if [[ "$OSTYPE" == "darwin"* ]]; then
            python "$SCRIPT_DIR/desktop_app.py" &
            sleep 0.5
            osascript -e 'tell application "System Events" to set frontmost of first process whose name is "Python" to true' 2>/dev/null || true
        else
            python "$SCRIPT_DIR/desktop_app.py"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|log|test|test-real|sync|dedupe|cleanup|gui}"
        echo ""
        echo "Background Daemon Commands:"
        echo "  start     - Start organizer as background daemon (PRODUCTION MODE)"
        echo "  stop      - Stop all file_organizer.py processes"
        echo "  restart   - Restart the background daemon"
        echo "  status    - Check if daemon is running"
        echo "  log       - Tail the log file (for background daemon only)"
        echo ""
        echo "Interactive Commands (see output in terminal):"
        echo "  test      - Run single scan (TEST MODE) - interactive"
        echo "  test-real - Run single scan (PRODUCTION MODE) - interactive"
        echo "  sync      - Synchronize folders only (PRODUCTION MODE) - interactive"
        echo "  dedupe    - Remove duplicates only (PRODUCTION MODE) - interactive"
        echo "  cleanup   - Remove broken/excluded symlinks from ~/organized"
        echo "  gui       - Launch desktop GUI application"
        echo ""
        echo "Note: Interactive commands show output directly. Background daemon"
        echo "      logs to ~/.file_organizer.log (use 'log' command to monitor)."
        exit 1
        ;;
esac

exit 0
