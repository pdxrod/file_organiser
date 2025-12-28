# File Organizer

> **⚠️ WARNING - USE AT YOUR OWN RISK ⚠️**
>
> This tool moves, copies, synchronizes, and can DELETE files across your system.
> - Always keep current backups
> - Start in **test mode** and review logs before using real data
> - Review your `config.yaml` carefully
> - Folder sync and deduplication can delete real files when misconfigured
> - No warranty is provided - see LICENSE

---

## Overview

File Organizer has three main capabilities:

- **Organization**: Scans your files and creates a tree of **soft links** under `output_base` (e.g. `~/organized/`), grouped by type, year, and discovered content categories. **Original files stay where they are.**
- **Folder Sync**: Bidirectionally synchronizes configured folder pairs (e.g. main drive ↔ external backup drive).
- **Duplicate Removal (Deduplication)**: Optionally finds and removes **duplicate real files** under configured `source_folders`.

There are two primary ways to use it:

- **Desktop app** (`./manage_organizer.sh gui`) – recommended; wraps all modes with a GUI.
- **Command line** (`python file_organizer.py ...`) – for advanced/terminal-heavy use.

---

## Modes (What Actually Happens)

### 🧪 Test Mode (default)

- **How to run (CLI):**
  - `python file_organizer.py --scan-once`
- **What it does:**
  - Uses a local `test/` directory and auto-created sample files.
  - Runs **one full organization cycle** then exits.
- **Files affected:**
  - Only under `test/` (created by `--create-test` or lazily on first run).
  - Real files outside `test/` are never touched.
- **Output:**
  - Soft-link tree under `test/organized/`.

### 🏭 Production Mode ("REAL")

- **How to run (single pass):**
  - `python file_organizer.py --REAL --scan-once`
- **How to run (daemon / continuous):**
  - `python file_organizer.py --REAL`  (desktop app can also start/stop this)
- **What it does in one full cycle (`run_full_cycle`):**
  1. Scans and organizes files (creates/updates soft links under `output_base`).
  2. Optionally synchronizes configured `sync_pairs` (bidirectional folder sync).
  3. Optionally runs **deduplication** (if enabled and configured).
  4. Optionally queues background backups (if enabled).
- **Files affected:**
  - Real files in the directories referenced by `source_folders`, `sync_pairs`, and `backup_directories`.
  - Soft links under `output_base` (e.g. `~/organized/`).

### 🔄 Sync Only

- **How to run:**
  - `python file_organizer.py --REAL --sync-only`
- **What it does:**
  - Skips organization and dedup.
  - Runs only **folder synchronization** for all `sync_pairs`.
- **Files affected:**
  - Real files in the folders referenced by `sync_pairs`.

### 🗑️ Dedupe Only

- **How to run:**
  - `python file_organizer.py --REAL --dedupe-only`
- **What it does (current implementation):**
  - Looks at the `source_folders` list in `config.yaml`.
  - Recursively scans those directories for **real files**.
  - Groups files by **content hash** (MD5) and removes duplicates, keeping only the newest copy in each group.
- **Important:**
  - This **operates on original files**, not on `~/organized` soft links.
  - It does **not** currently clean up duplicate soft links; it deletes duplicate real files under `source_folders`.

> If you want dedup to do nothing, either:
> - Leave `source_folders` unset/empty, or
> - Set `"enable_duplicate_detection": false`.

The README text in earlier versions said dedupe only touched `~/organized` soft links; that is **no longer accurate** for this implementation.

---

## Key Concepts

### Drives and Placeholders

In `config.yaml` you can define **drive shortcuts**:

```yaml
drives:
  MAIN_DRIVE: "/Users/yourname"
  EXTERNAL_DRIVE: "/Volumes/YourExternalDrive"
  PROTON_DRIVE: "MAIN_DRIVE/ProtonDrive"
  GOOGLE_DRIVE: "MAIN_DRIVE/GoogleDrive/MyFiles"
```

- Placeholders like `MAIN_DRIVE`, `GOOGLE_DRIVE`, etc. are resolved at startup.
- Nested references are supported, e.g. `PROTON_DRIVE` referencing `MAIN_DRIVE`.
- After resolution, the program rewrites:
  - `"MAIN_DRIVE/dev"` → `"/Users/yourname/dev"`
  - `"GOOGLE_DRIVE/Documents"` → `"/Users/yourname/GoogleDrive/MyFiles/Documents"`
- **After this phase the code should never see the literal string `MAIN_DRIVE` again** – only real paths.

### Sync Pairs (Folder Synchronization)

`sync_pairs` describe which folders should be synchronized. The current recommended format uses a `folders` array of two paths (order doesn't matter – sync is bidirectional):

```yaml
sync_pairs:
  - folders:
      - "MAIN_DRIVE/dev"
      - "EXTERNAL_DRIVE/dev"
  - folders:
      - "MAIN_DRIVE/Documents"
      - "GOOGLE_DRIVE/Documents"
```

- Each `folders` entry becomes two absolute paths after drive resolution.
- Sync logic (bidirectional):
  - If file exists only in A → copy A → B.
  - If file exists only in B → copy B → A.
  - If file exists in both and B is newer → copy B → A.
  - Otherwise → copy A → B.
- Exclusions: `exclude_patterns` are respected (e.g. `.git` folders are backed up and replaced with soft links, `node_modules`, `.tmp*`).

There is also an **old format** (`source`/`target`) still supported for backward compatibility, but new configs should prefer `folders`.

### Source Folders (Deduplication Scope)

`source_folders` are **only used for deduplication** in this implementation:

```yaml
source_folders:
  - "MAIN_DRIVE/Documents"
  - "MAIN_DRIVE/Pictures"
```

- These are resolved via the `drives` section the same way as `sync_pairs`.
- Dedup scans **all real files** under these directories (recursively, across drives if placeholders point there).
- For each group of identical files (by content hash):
  - It keeps the **newest** file (by modification time).
  - It **deletes** the older copies.

If `source_folders` is empty or missing, dedup logs:

> `No source_folders configured - skipping duplicate detection`

…and does nothing else.

### Organized Output (Soft Links)

The organizer writes soft links into `output_base` (default `~/organized` in production, `test/organized` in test mode):

- Links are grouped by type (`documents/`, `images/`, etc.), year (`2024/`, `2025/`), and discovered content categories (`python/`, `budget/`, etc.).
- Each entry is a **symlink** pointing back to the original file.
- The organizer **never moves** your original files as part of the organization step; it only creates/removes symlinks.

> **Important distinction:**
> - **Organization** step → manipulates soft links under `output_base`.
> - **Sync** and **dedup** → operate on real files under `sync_pairs` and `source_folders`.

---

## Configuration Summary

Core keys in `config.yaml`:

- **`drives`**: Drive shortcuts, can be nested and used in other paths.
- **`sync_pairs`**: Folder pairs to keep in sync (bidirectional).
- **`source_folders`**: Roots to scan for deduplication of real files.
- **`exclude_patterns`**: Names/patterns to skip (e.g. `.git`, `node_modules`, `.tmp*`).
- **`output_base`**: Root for the organized soft-link tree (e.g. `"~/organized"`).
- **`enable_content_analysis`**: Turn ML-based category discovery on/off.
- **`enable_folder_sync`**: Enable/disable running the sync step.
- **`enable_duplicate_detection`**: Enable/disable dedup across `source_folders`.
- **`backup_drive_path` / `backup_directories`**: Background backup destination and sources.

A minimal production config using drives and sync pairs might look like:

```yaml
drives:
  MAIN_DRIVE: "/Users/rod"
  EXTERNAL_DRIVE: "/Volumes/PASSPORT3"
  PROTON_DRIVE: "MAIN_DRIVE/ProtonDrive"
  GOOGLE_DRIVE: "MAIN_DRIVE/GoogleDrive/MyFiles"

sync_pairs:
  - folders:
      - "MAIN_DRIVE/dev"
      - "GOOGLE_DRIVE/dev"
  - folders:
      - "MAIN_DRIVE/Documents"
      - "PROTON_DRIVE/Documents"

exclude_patterns:
  - "node_modules"
  - "_build"
  - "deps"
  - "ebin"
  - ".git"
  - "__pycache__"
  - ".pytest_cache"
  - ".mypy_cache"
  - ".tox"
  - ".venv"
  - "venv"
  - "env"
  - "dist"
  - "build"
  - "target"
  - ".next"
  - ".cache"
  - ".parcel-cache"
  - "coverage"
  - ".nyc_output"
  - "elm-stuff"
  - ".elixir_ls"
  - ".stack-work"
  - "Photos Library.photoslibrary"
  - ".photoslibrary"
  - "iPhoto Library"
  - "Lightroom"
  - ".bundle"
  - "vendor"
  - "bundle"
  - "priv/static"
  - ".gradle"
  - ".m2"
  - "tmp/cache"
  - ".tmp*"
  - ".DS_Store"
  - "*.pyc"
  - "*.log"
  - ".Spotlight-V100"
  - ".TemporaryItems"
  - ".fseventsd"
  - ".DocumentRevisions-V100"

output_base: "~/organized"
enable_content_analysis: true
enable_folder_sync: true
enable_duplicate_detection: false
```

If you later want deduplication of **real files**, you would add for example:

```yaml
source_folders:
  - "MAIN_DRIVE/Documents"
  - "MAIN_DRIVE/Pictures"
enable_duplicate_detection: true
```

…and then run with `--REAL` (full cycle) or `--REAL --dedupe-only` (just dedup).

---

## CLI Reference (Current Behavior)

```bash
python file_organizer.py [OPTIONS]

Options:
  -R, --REAL           Run in PRODUCTION mode (default: TEST mode)
  --scan-once          Run a single organization/sync/dedupe cycle, then exit
  --create-test        Create test environment under ./test and exit
  --sync-only          Only synchronize folders (production mode)
  --dedupe-only        Only run deduplication (production mode)
  --config PATH        Use a custom config file (default: config.yaml)
```

Typical flows:

- **Safe test run:**
  - `python file_organizer.py --scan-once`
- **One-shot real run:**
  - `python file_organizer.py --REAL --scan-once`
- **Daemon (continuous real mode):**
  - `python file_organizer.py --REAL`
- **Just sync:**
  - `python file_organizer.py --REAL --sync-only`
- **Just dedupe (real files under source_folders):**
  - `python file_organizer.py --REAL --dedupe-only`

Logs are written to `~/.file_organizer.log`; the desktop app exposes them in a viewer, or you can use:

```bash
tail -f ~/.file_organizer.log
```

---

## Safety Checklist

Before running in production mode with real files:

- **Backups:** You have current backups of anything important.
- **Tested:** You have run at least one full cycle in test mode and reviewed `test/organized/`.
- **Config reviewed:** `config.yaml` is valid YAML (no tabs, proper indentation) and `drives`, `sync_pairs`, and (if used) `source_folders` point only to locations you are comfortable modifying.
- **Dedup clarity:** You understand that current dedup logic deletes **real files under `source_folders`**, not just soft links.
- **Logs monitored:** You know how to watch `~/.file_organizer.log` and stop the process if something looks wrong.

If any of the above is unclear, stay in test mode or run with `enable_duplicate_detection: false` and limited `sync_pairs` until you’re confident.
