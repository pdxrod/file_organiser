#!/usr/bin/env bash
set -euo pipefail

# ===== PATTERNS =====
# Folder-patterns: no wildcards → directory match
# File-patterns: contain * or ? → file match

EXCLUDE_PATTERNS=(
    # Node / JS
    "node_modules" ".next" ".nuxt" ".cache" ".parcel-cache" "dist" "build" "coverage"
    ".vite" ".yarn" ".pnp" ".pnp.js"

    # Python
    "__pycache__" ".pytest_cache" ".mypy_cache" ".tox" ".venv" "venv" "env"
    "*.pyc" "*.pyo" ".ruff_cache"

    # Ruby
    "vendor" ".bundle" "bundle" "tmp"

    # Elixir/Erlang
    "_build" "deps" "ebin" "priv/static" ".elixir_ls"

    # JVM
    "target" ".gradle" ".m2" "out" "classes"

    # Haskell
    ".stack-work" "dist-newstyle" "cabal-dev"

    # Rust
    "target"

    # C/C++
    "CMakeFiles" "*.o" "*.so" "*.a"

    # VCS ".git"
    ".hg" ".svn" ".cvs"

    # macOS
    ".DS_Store" ".TemporaryItems" ".fseventsd" "Icon?" "._*"

    # Logs
    "*.log" "log" "logs"

    # Misc
    "elm-stuff" ".nyc_output" ".vercel" ".svelte-kit" "tmp/cache"
    "Photos Library.photoslibrary" ".photoslibrary" "iPhoto Library" "Lightroom"
    "mongod.lock"
)

# ===== PROCESS ARGUMENTS =====
REAL_RUN=false
if [[ "${1:-}" == "-R" || "${1:-}" == "--real" ]]; then
    REAL_RUN=true
fi

# Split patterns: folders vs files
declare -a FOLDER_PATTERNS=()
declare -a FILE_PATTERNS=()

for p in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ "$p" == *"*"* || "$p" == *"?"* ]]; then
        FILE_PATTERNS+=("$p")
    else
        FOLDER_PATTERNS+=("$p")
    fi
done

# ===== Helper to empty a matched directory =====
empty_directory() {
    local dir="$1"
    if $REAL_RUN; then
        printf "Deleting everything inside directory: %s\n" "$dir"
        find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    else
        printf "Would delete everything inside directory: %s\n" "$dir"
    fi
}

# ===== WALK THE TREE =====
export LC_ALL=C

find . -depth -print0 | while IFS= read -r -d '' path; do
    # Skip root itself
    [[ "$path" == "." ]] && continue

    base="$(basename "$path")"

    # === Directory matches (no wildcards) ===
    if [[ -d "$path" ]]; then
        for dp in "${FOLDER_PATTERNS[@]}"; do
            if [[ "$base" == "$dp" ]]; then
                empty_directory "$path"
                # Do NOT recurse further into it — skip children
                break
            fi
        done
        continue
    fi

    # === File pattern matches (wildcards) ===
    if [[ -f "$path" ]]; then
        for fp in "${FILE_PATTERNS[@]}"; do
            if [[ "$base" == $fp ]]; then
                if $REAL_RUN; then
                    printf "Deleting file: %s\n" "$path"
                    rm -f "$path"
                else
                    printf "Would delete file: %s\n" "$path"
                fi
                break
            fi
        done
    fi
done

if ! $REAL_RUN; then
    echo
    echo "Dry run complete. Use --real or -R to actually delete."
fi

