# File Organizer

> **⚠️ WARNING - USE AT YOUR OWN RISK ⚠️**
>
> This tool moves, copies, synchronizes, and can DELETE files across your entire file system.
> - Test thoroughly in TEST MODE before using production mode
> - Always maintain current backups before running in production
> - Review your `organizer_config.json` carefully
> - The `--delete` flag in rsync will remove files from target that don't exist in source
> - No warranty is provided - see LICENSE file
>
> **YOU are responsible for any data loss. Use with caution.**

---

## 🖥️ Desktop App (Recommended)

**The easiest way to use File Organizer is with the desktop app!**

### Launch the Desktop App
```bash
./manage_organizer.sh gui
```

The desktop app provides:
- **Visual controls** - Start/stop the organizer with simple buttons
- **Mode selection** - Choose test mode, production mode, daemon mode, etc. with checkboxes
- **Live log viewer** - See what's happening in real-time
- **File tree browser** - Browse your organized files visually
- **Cross-platform** - Works on Mac, Linux, and Windows

### What the Desktop App Does

The desktop app gives you a friendly interface to:
- **Test Mode** - Safely test the organizer without making changes
- **Production Mode** - Run the full organizer with your real files
- **Daemon Mode** - Run continuously in the background
- **Sync Only** - Just synchronize folders between drives
- **Dedupe Only** - Just find and remove duplicate files

---

## 🚀 Quick Start (Desktop App)

### Requirements

- **Python 3.13** (tested) or Python 3.8+
- pip package manager

### 1. Install Dependencies

**Python packages:**
```bash
pip install -r requirements.txt
```

**System tools (required for OCR and video/PDF processing):**
```bash
# macOS
brew install tesseract poppler

# Linux
apt-get install tesseract-ocr poppler-utils
```

### 2. Launch the Desktop App

```bash
# Start the desktop app
./manage_organizer.sh gui
```

The desktop app will open with a friendly interface where you can:
- Select **Test Mode** to safely experiment
- Choose **Production Mode** when ready for real files
- Pick **Daemon Mode** to run continuously
- Use **Sync Only** or **Dedupe Only** for specific tasks

### 3. Configure for Production (Optional)

If you want to use the desktop app with your real files, first create a configuration:

```bash
# Copy the template
cp organizer_config.template.json organizer_config.json

# Edit with your actual paths
nano organizer_config.json  # or use your favorite editor
```

Edit the `"drives"` section with your actual paths:

```json
{
    "drives": {
        "MAIN_DRIVE": "/Users/yourname",
        "GOOGLE_DRIVE": "/Users/yourname/Google Drive",
        "PROTON_DRIVE": "/Users/yourname/ProtonDrive"
    },
    "source_folders": [
        "MAIN_DRIVE/Documents",
        "MAIN_DRIVE/Pictures"
    ]
}
```

---

## 💻 Command Line Interface (For Advanced Users)

**Note: The desktop app is recommended for most users. The command line is for advanced users who prefer terminal-based control.**

### Test Mode (Safe)
```bash
# Single scan of test/ directory
python file_organizer.py --scan-once

# Check the results
ls -la test/organized/
```

### Production Mode
```bash
# Single scan (safe, review first)
python file_organizer.py --REAL --scan-once

# Start daemon mode (runs continuously)
python file_organizer.py --REAL
```

### Management Script
```bash
# Background daemon commands
./manage_organizer.sh start    # Start daemon
./manage_organizer.sh stop     # Stop daemon
./manage_organizer.sh status   # Check status
./manage_organizer.sh log      # View logs

# Interactive commands
./manage_organizer.sh test     # Test mode
./manage_organizer.sh test-real # Production mode
./manage_organizer.sh sync     # Sync only
./manage_organizer.sh dedupe   # Dedupe only
```

## 🎯 Key Features

### 1. **Dynamic Content Discovery** ⭐

The organizer **learns categories from your files**.

- Analyzes filenames, folder names, and content
- Discovers patterns and creates categories automatically
- Only creates categories with enough matching files
- No irrelevant categories

**Example:**
- You have 15 files about "budget" → Creates `/budget/` category
- You have 127 Python files → Creates `/python/` category  
- You have 0 files about "fishing" → No `/fishing/` category

### 2. **Smart Year Classification**

- Filename dates override file metadata
- `20240101-report.txt` goes in `/2024/` even if created in 2025
- Supports formats: YYYYMMDD, YYYY-MM-DD, YYYY, and even DD-MM-YY
- Falls back to file creation/modification dates

### 3. **Test & Production Modes**

- **Test mode (default)**: Safe testing with auto-created test folders
- **Production mode (--REAL)**: Your actual file system
- Easy to experiment without risk

### 4. **File Type Organization**

Automatically organizes by file type:
- `documents/` - .txt, .doc, .docx, .pdf, .rtf, .odt
- `images/` - .jpg, .png, .gif, .bmp, .tiff, .webp
- `videos/` - .mp4, .avi, .mov, .mkv, .wmv
- `audio/` - .mp3, .wav, .flac, .aac, .ogg
- `code/` - .py, .js, .html, .css, .java, .cpp

**Text Extraction & Content Analysis:**
- `.txt` - Plain text ✅
- `.docx` - Microsoft Word (python-docx) ✅
- `.rtf` - Rich Text Format (striprtf) ✅
- `.odt` - OpenDocument Text (odfpy) ✅
- `.doc` - Old Word format (basic text extraction, limited) ⚠️
- `.pdf` - PDF files (PyPDF2) ✅
  - **Scanned PDFs**: Auto-detects and uses OCR if no text found ✨
- **Images** - Advanced OCR and object recognition ✨ **NEW!**
  - **EasyOCR**: Reads text in photos, artistic fonts, low contrast (better than Tesseract!)
  - **Tesseract**: Fallback OCR with multiple modes
  - **CLIP** (optional): Recognizes objects in photos (fish, people, buildings, etc.)
  - Formats: `.jpg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`
- **Videos** - Frame-by-frame OCR text extraction ✨
  - Formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.m4v`
  - Samples frames every 10 seconds, extracts visible text

**System Requirements:**
- `tesseract` OCR engine: `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux)
- `poppler-utils` for PDF→image: `brew install poppler` (macOS) or `apt install poppler-utils` (Linux)

**AI Features (Optional):**
- **EasyOCR**: Enabled by default, works on CPU (slower than Tesseract but more accurate)
- **CLIP**: Disabled by default (very slow on CPU ~30min first load). Enable with `"use_clip": true` in config if you have a GPU

### 5. **Advanced Features** (Production Mode)

- **Folder Synchronization**: Keep folder pairs in sync with rsync
- **Concurrent Chunked Sync**: Large folders sync in parallel chunks for faster progress
- **Cloud Storage Optimized**: Special flags for Google Drive, ProtonDrive, and other FUSE mounts
- **Duplicate Detection**: Find and remove duplicate files
- **Git Version Control**: Track all changes automatically
- **Background Backup**: Backup to external or cloud drives with retry logic
- **Robust Error Handling**: Graceful handling of flaky drives and timeouts

## 📁 Expected Output

After running in test mode:

```
test/
├── foo/
├── bar/
├── baz/
└── organized/
    ├── 2024/              # Files from 2024
    ├── 2025/              # Files from 2025
    ├── documents/         # All document files
    ├── images/            # All image files
    ├── davis/             # Discovered: Miles Davis (from music files)
    ├── ella/              # Discovered: Ella Fitzgerald
    ├── ella-fitzgerald/   # Discovered: Full name (bigram)
    ├── existence/         # Discovered: Philosophy content
    ├── fitzgerald/        # Discovered: From music collection
    ├── life/              # Discovered: Philosophy theme
    ├── miles/             # Discovered: Miles Davis
    ├── miles-davis/       # Discovered: Full name (bigram)
    ├── music/             # Discovered: Music-related content
    ├── notes/             # Discovered: Notes files
    ├── peggy/             # Discovered: Peggy Lee (OCR from image + text)
    ├── people/            # Discovered: CLIP vision detected people
    └── something/         # Discovered: From filenames
```

Each folder contains soft links to the actual files wherever they are. Categories are discovered automatically by analyzing file content, filenames, and even OCR text from images!

## 🧠 How Dynamic Content Discovery Works

### Phase 1: Learning (During Scan)
1. Scans all your files
2. Extracts keywords from filenames and folder names
3. Extracts keywords from file content (if enabled)
4. Counts keyword frequencies across all files

### Phase 2: Discovery (After Scan)
1. Finds keywords that appear ≥ 3 times (configurable)
2. Creates categories for significant keywords
3. Only creates category if ≥ 5 files match (configurable)
4. Keeps top 50 categories by frequency (configurable)

### Phase 3: Organization
1. Creates soft link folders for discovered categories
2. Adds soft links to all matching files
3. Saves discovered categories to JSON for review

## 📊 Review Discovered Categories

After running:
```bash
cat ~/.file_organizer_discovered_categories.json
```

Example output:
```json
{
  "python": {
    "file_count": 127,
    "sample_files": ["/path/to/script.py", ...]
  },
  "budget": {
    "file_count": 15,
    "sample_files": ["/path/to/2024-budget.xlsx", ...]
  }
}
```

## ⚙️ Configuration

### Basic Configuration (Required for Production Mode)

Edit `organizer_config.json`:

```json
{
    "scan_interval": 3600,
    "source_folders": [
        "/path/to/documents",
        "/path/to/pictures"
    ],
    "exclude_folders": [
        "/path/to/backup",
        "/path/to/.git"
    ],
    "output_base": "/path/to/organized",
    "enable_content_analysis": true,
    "enable_duplicate_detection": false,
    "enable_folder_sync": false,
    "enable_git_tracking": false,
    "enable_proton_backup": false
}
```

### Advanced Configuration

```json
{
    "ml_content_analysis": {
        "enabled": true,
        "min_keyword_frequency": 8,
        "min_category_size": 5,
        "max_categories": 250,
        "min_word_length": 5,
        "stop_words_enabled": true
    },
    "sync_pairs": [
        {
            "source": "/path/to/source",
            "target": "/path/to/target"
        }
    ],
    "backup_directories": [
        "/path/to/backup1",
        "/path/to/backup2"
    ],
    "backup_drive_path": "/path/to/backupdrive",
    "use_rsync": true,
    "rsync_checksum_mode": "timestamp",
    "rsync_size_only": true,
    "rsync_additional_args": [
        "--omit-dir-times",
        "--no-perms", 
        "--no-group",
        "--no-owner",
        "--delete-after"
    ],
    "sync_chunk_subfolders": 10,
    "sync_chunk_concurrency": 3,
    "sync_timeout_minutes": 180
}
```

### Key Settings:

- **`source_folders`**: Directories to scan
- **`exclude_folders`**: Directories to skip (avoid recursion!)
- **`output_base`**: Where to create soft link folders
- **`enable_content_analysis`**: Enable/disable ML content discovery
- **`ml_content_analysis`**: Tune category discovery thresholds
- **`use_rsync`**: Use rsync for fast folder synchronization
- **`rsync_size_only`**: Only compare file sizes (faster for cloud storage)
- **`rsync_additional_args`**: Extra rsync flags to reduce FUSE overhead
- **`sync_chunk_concurrency`**: Number of parallel sync operations
- **`sync_timeout_minutes`**: Timeout for large folder syncs

## 🔧 Tuning Category Discovery

**If too many categories:**
```json
"ml_content_analysis": {
    "min_keyword_frequency": 5,    // Increase
    "min_category_size": 10,        // Increase
    "max_categories": 30            // Decrease
}
```

**If too few categories:**
```json
"ml_content_analysis": {
    "min_keyword_frequency": 2,     // Decrease
    "min_category_size": 3,          // Decrease
    "max_categories": 250,           // Increase (current default)
    "min_word_length": 4             // Allow shorter words
}
```

## 🎮 Usage

### Test Mode

```bash
# Create test environment and run
python file_organizer.py --create-test
python file_organizer.py --scan-once

# Or just run (auto-creates test if needed)
python file_organizer.py --scan-once

# Check results
ls -la test/organized/
```

### Production Mode

```bash
# Single scan (safe, review first)
python file_organizer.py --REAL --scan-once

# Review results
tail -100 ~/.file_organizer.log

# Start daemon (runs continuously)
python file_organizer.py --REAL

# Advanced: specific operations
python file_organizer.py --REAL --sync-only      # Only sync folders
python file_organizer.py --REAL --dedupe-only    # Only remove duplicates
```

### Command Line Options

```bash
python file_organizer.py [OPTIONS]

Options:
  -R, --REAL           Run in PRODUCTION mode (default: TEST mode)
  --scan-once          Run single scan instead of daemon
  --create-test        Create test environment and exit
  --sync-only          Only synchronize folders (production mode)
  --dedupe-only        Only remove duplicates (production mode)
  --config PATH        Custom config file path
```

## 📝 Logging

All activity is logged to `~/.file_organizer.log`

Monitor in real-time:
```bash
tail -f ~/.file_organizer.log
```

Check for errors:
```bash
grep ERROR ~/.file_organizer.log
```

## 🛡️ Safety Features

1. **Soft Links Only** - Original files never moved or modified
2. **Test Mode** - Safe experimentation with isolated test folders
3. **Git Tracking** - Optional version control for all changes
4. **Comprehensive Logging** - Know exactly what happened
5. **Exclude Folders** - Prevent recursion and protect system folders
6. **Graceful Error Handling** - Continues on individual file failures

## 🆘 Troubleshooting

### Nothing is happening

```bash
# Check the log
tail -100 ~/.file_organizer.log

# Look for errors
grep ERROR ~/.file_organizer.log
```

### Permission denied errors

Some directories may require elevated permissions:
```bash
sudo python file_organizer.py --REAL --scan-once
```

### Too many/few categories

Adjust thresholds in `organizer_config.json` under `ml_content_analysis`:
- Increase `min_keyword_frequency` for fewer categories
- Decrease `min_category_size` for more categories
- Adjust `max_categories` to limit total

### No categories discovered

- Check that `enable_content_analysis` is true
- Lower `min_keyword_frequency` and `min_category_size`
- Verify files have readable content
- Check log for content analysis errors

## 📚 Examples

### Example 1: First Time Test

```bash
$ python file_organizer.py --scan-once
======================================================================
TEST MODE - Operating on test/ directory
======================================================================

$ ls test/organized/
2024/  2025/  documents/  images/  insurance/  fishing/
```

### Example 2: Production Mode

```bash
$ python file_organizer.py --REAL --scan-once
======================================================================
PRODUCTION MODE - Operating on your entire file system
======================================================================

$ cat ~/.file_organizer_discovered_categories.json
{
  "python": {"file_count": 127},
  "javascript": {"file_count": 85},
  "budget": {"file_count": 15}
}
```

### Example 3: Continuous Monitoring

```bash
$ python file_organizer.py --REAL &
$ tail -f ~/.file_organizer.log
```

## ⚠️ Before Production Use

**Essential Steps:**

1. ✅ Run in test mode first
2. ✅ Review discovered categories
3. ✅ Configure paths for your system
4. ✅ Start with a small subset of folders
5. ✅ Have backups of critical data
6. ✅ Monitor logs for first few cycles
7. ✅ Understand how to stop it (Ctrl+C or kill)

## 🎓 Advanced Features

### Folder Synchronization

Keep two folders in sync with optimized rsync:

```json
"enable_folder_sync": true,
"use_rsync": true,
"rsync_checksum_mode": "timestamp",
"rsync_size_only": true,
"rsync_additional_args": [
    "--omit-dir-times",
    "--no-perms",
    "--no-group", 
    "--no-owner",
    "--delete-after"
],
"sync_pairs": [
    {
        "source": "/source/folder",
        "target": "/target/folder"
    }
],
"sync_chunk_subfolders": 10,
"sync_chunk_concurrency": 3,
"sync_timeout_minutes": 180
```

**Performance Notes:**
- `rsync_size_only: true` - Much faster for cloud storage (Google Drive, ProtonDrive)
- `rsync_additional_args` - Reduces FUSE metadata overhead
- `sync_chunk_concurrency: 3` - Sync multiple subfolders in parallel
- `sync_timeout_minutes: 180` - 3-hour timeout for large folders

### Duplicate Detection

Find and remove duplicates (keeps newest):

```json
"enable_duplicate_detection": true
```

### Git Version Control

Track all changes with Git:

```json
"enable_git_tracking": true,
"git_user": "Your Name",
"git_email": "your.email@example.com"
```

### Background Backup

Background backup with retry logic and cloud storage support:

```json
"enable_background_backup": true,
"backup_drive_path": "/path/to/backupdrive",
"backup_directories": [
    "/path/to/important/folder"
]
```

**Supported Backup Targets:**
- External drives (USB, Thunderbolt)
- Cloud storage mounts (Google Drive, ProtonDrive, Dropbox)
- Network drives (SMB, NFS)
- Any mounted filesystem

## 🎉 What Makes This Special

1. **No Hardcoded Categories** - Learns from YOUR files
2. **Smart Date Handling** - Filename dates override metadata
3. **Test & Production Modes** - Safe experimentation
4. **Cloud Storage Optimized** - Special handling for Google Drive, ProtonDrive, etc.
5. **Concurrent Sync** - Large folders sync in parallel chunks
6. **ML-Powered** - Discovers patterns in your files
7. **Comprehensive** - Handles edge cases gracefully
8. **Portable** - Works anywhere, no hardcoded paths

## 📞 Support

**Before asking for help:**
1. **Use the desktop app** - It shows logs and status visually
2. Check the logs: `tail -100 ~/.file_organizer.log` or use the desktop app's log viewer
3. Review configuration: `cat organizer_config.json`
4. Test mode first: Use "Test Mode" in the desktop app or `python file_organizer.py --scan-once`
5. Verify paths are absolute and exist

## 📜 License

MIT License - Free to use and modify

---

## 🚦 Getting Started Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Run test mode (`python file_organizer.py --scan-once`)
- [ ] Review test results (`ls test/organized/`)
- [ ] Check discovered categories (`cat ~/.file_organizer_discovered_categories.json`)
- [ ] Edit config for your system (`organizer_config.json`)
- [ ] Test with one small folder first
- [ ] Review logs for errors (`tail ~/.file_organizer.log`)
- [ ] Gradually expand to more folders
- [ ] Consider enabling advanced features

**Remember:** Start with test mode, then start small in production mode!

```bash
# Safe way to start
python file_organizer.py --scan-once              # Test mode
python file_organizer.py --REAL --scan-once        # Production (review first!)
```

**Your files, your categories, your way.** 🚀
