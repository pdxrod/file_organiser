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

## Getting Started: Setting Up config.yaml

`config.yaml` tells the organizer where your files live, which folders to keep in sync, and how to behave. You must have one before the program will run.

### Step 1 — Let the app create a starter config

The easiest way to get a config is simply to run the app. If no `config.yaml` exists, it creates one automatically:

```bash
./manage_organizer.sh test
```

You will see something like:

```
CONFIG FILE CREATED — PLEASE EDIT BEFORE RUNNING AGAIN
======================================================================

No 'config.yaml' was found, so a starter file has been created.

MAIN_DRIVE is set to '/Users/janedoe' which does not exist on this system.

Next steps:
  1. Open config.yaml in a text editor.
  2. Set MAIN_DRIVE to your actual home directory ...
```

### Step 1 (alternative) — Copy the template manually

If you prefer to start from the full annotated template instead:

```bash
cp config_template.yaml config.yaml
```

`config_template.yaml` contains every available option with explanatory comments. The auto-generated `config.yaml` is a shorter, cleaner version — either works.

---

### Step 2 — Edit config.yaml

Open it in any text editor (TextEdit on Mac, Notepad on Windows, gedit on Linux). The file is in **YAML format**. YAML is simple but strict — the most common mistake is using TAB characters for indentation. **Always use spaces, never tabs.** Two spaces per indent level is standard.

#### The only line you MUST change

```yaml
drives:
  MAIN_DRIVE: "/Users/janedoe"   # ← change this
```

Replace `/Users/janedoe` with your actual home directory:

| Operating system | Typical value |
|-----------------|---------------|
| macOS | `/Users/yourname` |
| Linux | `/home/yourname` |
| Windows | `C:/Users/yourname` |

To find your home directory:
- **macOS / Linux:** open a terminal and type `echo $HOME`
- **Windows:** open PowerShell and type `echo $HOME`

#### Drives that don't apply to you — leave or delete them

The starter config includes example entries for external drives, cloud drives, Linux, and Windows. If you don't have an external drive right now, that's fine — the app will warn you that it doesn't exist and skip it. You can remove lines you don't need, or fill them in later.

```yaml
  EXTERNAL_DRIVE: "/Volumes/ExternalDrive"   # delete if you have no external drive
  PROTON_DRIVE: "MAIN_DRIVE/ProtonDrive"     # delete if you don't use ProtonDrive
  GOOGLE_DRIVE: "MAIN_DRIVE/GoogleDrive/MyFiles"
  LINUX_DRIVE: "/home/janedoe"               # delete if you're not on Linux
  WINDOWS_C: "C:/Users/janedoe"             # delete if you're not on Windows
  WINDOWS_D: "D:/"
```

Note that `PROTON_DRIVE: "MAIN_DRIVE/ProtonDrive"` references another drive by name — the app resolves this automatically, so you don't need to repeat the full path.

#### Sync pairs — which folders to keep in sync

`sync_pairs` is the heart of the config. Each pair names two folders that should stay identical. Files are copied in whichever direction is needed:

```yaml
sync_pairs:
  - folders:
      - "MAIN_DRIVE/Documents"
      - "PROTON_DRIVE/Documents"
```

This means: keep `~/Documents` and `~/ProtonDrive/Documents` in sync. If you add a file to either one, it will appear in the other on the next run.

To add more pairs, copy the block:

```yaml
sync_pairs:
  - folders:
      - "MAIN_DRIVE/Documents"
      - "PROTON_DRIVE/Documents"
  - folders:
      - "MAIN_DRIVE/Pictures"
      - "EXTERNAL_DRIVE/Pictures"
```

**Important:** each pair must have exactly two folders. The indentation must match exactly — two spaces before `folders`, four spaces before the `- "..."` lines.

#### One-way backups

`one_way_pairs` works the same way but only copies from the first folder to the second (never backwards). Use this for backups where you don't want changes on the backup to come back:

```yaml
one_way_pairs:
  - folders:
      - "MAIN_DRIVE/Music"
      - "PROTON_DRIVE/Music"
```

#### If you're not sure — start minimal

The safest first config just sets your home directory and leaves `sync_pairs` empty or with one low-risk pair. You can always add more later:

```yaml
drives:
  MAIN_DRIVE: "/Users/yourname"

sync_pairs:
  - folders:
      - "MAIN_DRIVE/Documents"
      - "MAIN_DRIVE/Documents-backup"

output_base: "~/organized"
enable_folder_sync: true
enable_duplicate_detection: false
```

---

### Step 3 — Test before going live

Always run in test mode first. It only touches the local `test/` directory — your real files are never affected:

```bash
./manage_organizer.sh test
```

Check the output for warnings or errors. When it finishes cleanly, look inside `test/organized/` to see what the soft-link tree would look like for your real files.

### Step 4 — Run for real

```bash
./manage_organizer.sh start    # starts as a background daemon
./manage_organizer.sh log      # watch what it's doing
./manage_organizer.sh stop     # stop it
```

---

### YAML common mistakes

| Mistake | Example of wrong | Correct |
|---------|-----------------|---------|
| Using a tab for indent | `→MAIN_DRIVE: ...` (tab character) | `  MAIN_DRIVE: ...` (two spaces) |
| Missing quotes around paths with spaces | `MAIN_DRIVE: /Users/Jane Doe` | `MAIN_DRIVE: "/Users/Jane Doe"` |
| Wrong number of spaces | `- folders:` then `    - "..."` (4 spaces) vs `      - "..."` (6 spaces) | be consistent — always 6 spaces before `- "..."` under `folders` |
| Colon inside a value without quotes | `MAIN_DRIVE: C:\Users\jane` | `MAIN_DRIVE: "C:/Users/jane"` |

If the app says your YAML is invalid, paste the contents of `config.yaml` into [yamllint.com](https://www.yamllint.com/) to find the exact error — it highlights the problem line.

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

### `~/organized` vs `~/organised` — what's the difference?

These are **two separate directories** with different purposes:

| Directory | Purpose | Who needs it |
|-----------|---------|--------------|
| `~/organized` | Soft-link tree of your files, browsable by type, year, and topic | Everyone |
| `~/organised` | Backup copies of development folders (`.git`, `__pycache__`, `.venv`, etc.) replaced with soft links during sync | Software developers only |

**`~/organized`** is set by the `output_base` key in `config.yaml`. It gives you a clean, browsable view of all your files without moving anything. You can safely delete it at any time — the organizer will rebuild it on the next run.

**`~/organised`** (British spelling, set by `softlink_backup_base`) is only relevant if you do software development. When the organizer encounters folders that match `softlink_folder_patterns` (like `.git`, `__pycache__`, `venv`), it moves their contents here and replaces the original folder with a soft link. This stops those bulky development folders from being copied wholesale during sync operations.

> **If you are not a software developer, you can ignore `~/organised` entirely.** It will either stay empty or never be created. The `softlink_backup_base` setting has no effect unless you have folders matching `softlink_folder_patterns` in your source directories.

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
