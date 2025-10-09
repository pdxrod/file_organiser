#!/usr/bin/env python3
"""
File Organizer using AI/ML

Unified version with test and production modes:
- DEFAULT (no args): Runs in TEST MODE with foo/bar/baz folders
- With --REAL or -R: Runs in PRODUCTION MODE on your entire file system

Features:
- Dynamic content discovery (learns categories from YOUR files)
- Multi-volume support across file systems
- Folder synchronization with duplicate removal
- Git version control integration (with proper user configuration)
- Background backup to remote drives
- Soft link preservation
- Notification system for offline volumes
- Robust error handling for flaky volumes
"""

import os
import sys
import time
import hashlib
import logging
import argparse
import signal
import threading
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import mimetypes
import json
import re
from collections import defaultdict
import queue

# Third-party imports for AI/ML content analysis
try:
    from PIL import Image
    import pytesseract
    import cv2  # OpenCV for video processing
except ImportError:
    Image = None
    pytesseract = None
    cv2 = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    import easyocr  # Better OCR for artistic text
except ImportError:
    easyocr = None

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError:
    CLIPProcessor = None
    CLIPModel = None
    torch = None

try:
    import docx
    import PyPDF2
except ImportError:
    docx = None
    PyPDF2 = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None

try:
    from odf import text as odf_text
    from odf.opendocument import load as odf_load
except ImportError:
    odf_text = None
    odf_load = None


GITIGNORE_TEMPLATE = """.fseventsd/
._.DS_Store
z.exs
config/z.exs

_build
deps
.elixir_ls
*.beam
erl_crash.dump
.compile.elixir

public/assets
node_modules
assets/node_modules
priv/static/js/app.js
priv/static/js/app.js.map

*.log
pkg/
log/*
tmp/*
.idea
*~
*\\~
*.raw
*.dta
*.~lock*
.lock
.*.swp
.DS_Store
.vscode/
"""


class DynamicContentAnalyzer:
    """
    Dynamically discovers content categories by analyzing files and folders.
    Uses ML/NLP techniques to extract meaningful keywords and topics.
    """
    
    def __init__(self, logger: logging.Logger, config: Dict):
        self.logger = logger
        self.config = config
        self.ml_config = config.get('ml_content_analysis', {})
        
        # Dynamic learning
        self.keyword_frequencies = defaultdict(int)
        self.file_keywords = {}  # file_path -> set of keywords
        self.discovered_categories = {}  # category_name -> set of keywords
        
        # Common stop words to ignore
        self.stop_words = self._get_stop_words() if self.ml_config.get('stop_words_enabled', True) else set()
        
        # Minimum thresholds
        self.min_keyword_freq = self.ml_config.get('min_keyword_frequency', 3)
        self.min_category_size = self.ml_config.get('min_category_size', 5)
        self.max_categories = self.ml_config.get('max_categories', 50)
        
        # AI models (lazy-loaded)
        self._easyocr_reader = None
        self._clip_model = None
        self._clip_processor = None
    
    def get_easyocr_reader(self):
        """Lazy-load EasyOCR reader."""
        if self._easyocr_reader is None and easyocr:
            try:
                self.logger.info("Loading EasyOCR model (first time only)...")
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                self.logger.info("EasyOCR loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load EasyOCR: {e}")
        return self._easyocr_reader
    
    def get_clip_model(self):
        """Lazy-load CLIP model."""
        if self._clip_model is None and CLIPModel and CLIPProcessor:
            try:
                self.logger.info("Loading CLIP model (first time only)...")
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.logger.info("CLIP loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load CLIP: {e}")
        return self._clip_model, self._clip_processor
    
    def _get_stop_words(self) -> Set[str]:
        """Get common English stop words."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us',
            'them', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'once', 'here', 'there', 'then',
            'now', 'get', 'got', 'one', 'two', 'file', 'document', 'folder', 'page',
            'com', 'www', 'http', 'https', 'org', 'net', 'pdf', 'doc', 'docx', 'txt'
        }
    
    def extract_keywords_from_text(self, text: str) -> Set[str]:
        """
        Extract meaningful keywords from text using NLP techniques.
        """
        if not text:
            return set()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s-]', ' ', text)
        
        # Split into words
        words = text.split()
        
        keywords = set()
        for word in words:
            # Skip if too short, too long, or is a stop word
            if len(word) < 3 or len(word) > 20 or word in self.stop_words:
                continue
            
            # Skip numbers
            if word.isdigit():
                continue
            
            keywords.add(word)
        
        # Also extract multi-word phrases (bigrams)
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in self.stop_words and w2 not in self.stop_words:
                if len(w1) >= 3 and len(w2) >= 3:
                    phrase = f"{w1}_{w2}"
                    keywords.add(phrase)
        
        return keywords
    
    def extract_keywords_from_path(self, path: Path) -> Set[str]:
        """
        Extract keywords from file and folder names in the path.
        Only extracts from filename and immediate parent folder, not full path.
        """
        keywords = set()
        
        # Extract from filename (without extension)
        filename = path.stem
        keywords.update(self.extract_keywords_from_text(filename))
        
        # Extract only from immediate parent folder (not entire path)
        # This avoids noise like 'Users', 'rod', 'dev', etc.
        if path.parent and path.parent.name not in ['/', '.', 'test']:
            parent_name = path.parent.name
            keywords.update(self.extract_keywords_from_text(parent_name))
        
        return keywords
    
    def analyze_file(self, file_path: Path, content: str = None) -> Set[str]:
        """
        Analyze a file and extract all relevant keywords.
        Combines path analysis with content analysis.
        """
        keywords = set()
        
        # Keywords from path
        keywords.update(self.extract_keywords_from_path(file_path))
        
        # Keywords from content
        if content:
            keywords.update(self.extract_keywords_from_text(content))
        
        # Update global frequency counts
        for keyword in keywords:
            self.keyword_frequencies[keyword] += 1
        
        # Store keywords for this file
        self.file_keywords[str(file_path)] = keywords
        
        return keywords
    
    def discover_categories(self) -> Dict[str, Set[str]]:
        """
        Discover meaningful categories based on keyword frequencies.
        Returns a dictionary of category_name -> representative files.
        """
        # Find keywords that appear frequently enough
        significant_keywords = {
            keyword for keyword, freq in self.keyword_frequencies.items()
            if freq >= self.min_keyword_freq
        }
        
        if not significant_keywords:
            self.logger.info("No significant keywords found for category discovery")
            return {}
        
        # Sort by frequency
        sorted_keywords = sorted(
            significant_keywords,
            key=lambda k: self.keyword_frequencies[k],
            reverse=True
        )[:self.max_categories]
        
        # Create categories for top keywords
        categories = {}
        for keyword in sorted_keywords:
            # Find files that have this keyword
            matching_files = {
                file_path for file_path, file_keywords in self.file_keywords.items()
                if keyword in file_keywords
            }
            
            # Only create category if enough files match
            if len(matching_files) >= self.min_category_size:
                # Clean up keyword for category name (remove underscores, make readable)
                category_name = keyword.replace('_', '-')
                categories[category_name] = matching_files
                self.logger.info(
                    f"Discovered category '{category_name}' with {len(matching_files)} files "
                    f"(keyword frequency: {self.keyword_frequencies[keyword]})"
                )
        
        self.discovered_categories = categories
        return categories
    
    def get_file_categories(self, file_path: Path) -> Set[str]:
        """
        Get discovered categories that this file belongs to.
        """
        categories = set()
        file_path_str = str(file_path)
        
        for category_name, files in self.discovered_categories.items():
            if file_path_str in files:
                categories.add(category_name)
        
        return categories
    
    def save_discovered_categories(self, output_path: Path):
        """
        Save discovered categories to a JSON file for review.
        """
        try:
            categories_summary = {
                category: {
                    'file_count': len(files),
                    'sample_files': list(files)[:10]  # Save first 10 as samples
                }
                for category, files in self.discovered_categories.items()
            }
            
            with open(output_path, 'w') as f:
                json.dump(categories_summary, f, indent=2)
            
            self.logger.info(f"Saved discovered categories to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save discovered categories: {e}")


class GitManager:
    """Manages Git operations for version control."""
    
    def __init__(self, logger: logging.Logger, config: Dict = None):
        self.logger = logger
        self.config = config or {}
        
        # Get git user from config or try to get from git global config
        self.default_git_user = self.config.get('git_user', self._get_git_global_user())
        self.default_git_email = self.config.get('git_email', self._get_git_global_email())
    
    def _get_git_global_user(self) -> str:
        """Get git user from global config."""
        try:
            result = subprocess.run(
                ['git', 'config', '--global', 'user.name'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip() or "UNCONFIGURED"
        except:
            return "UNCONFIGURED"
    
    def _get_git_global_email(self) -> str:
        """Get git email from global config."""
        try:
            result = subprocess.run(
                ['git', 'config', '--global', 'user.email'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip() or "UNCONFIGURED"
        except:
            return "UNCONFIGURED"
    
    def is_git_installed(self) -> bool:
        """Check if git is installed."""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def has_git_repo(self, path: Path) -> bool:
        """Check if directory has a .git folder."""
        return (path / '.git').exists()
    
    def configure_git_user(self, path: Path) -> None:
        """
        Configure git user for the repository.
        Skips folders listed in git_exclude_folders config.
        """
        try:
            # Check if path should be excluded from git config
            path_str = str(path).lower()
            exclude_folders = self.config.get('git_exclude_folders', [])
            
            for exclude in exclude_folders:
                if exclude.lower() in path_str:
                    self.logger.info(f"Skipping git user config for excluded folder: {path}")
                    return
            
            # Don't configure if user/email are unconfigured
            if 'UNCONFIGURED' in self.default_git_user or 'UNCONFIGURED' in self.default_git_email:
                self.logger.debug(f"Git user not configured - skipping git config for {path}")
                return
            
            # Set local git config for this repo
            subprocess.run(
                ['git', 'config', 'user.name', self.default_git_user],
                cwd=path,
                capture_output=True,
                check=True
            )
            subprocess.run(
                ['git', 'config', 'user.email', self.default_git_email],
                cwd=path,
                capture_output=True,
                check=True
            )
            self.logger.info(f"Configured git user for {path}: {self.default_git_user} <{self.default_git_email}>")
        except Exception as e:
            self.logger.warning(f"Failed to configure git user for {path}: {e}")
    
    def init_repo(self, path: Path) -> bool:
        """Initialize a new git repository."""
        try:
            subprocess.run(['git', 'init'], cwd=path, capture_output=True, check=True)
            self.logger.info(f"Initialized git repository in {path}")
            
            # Configure git user
            self.configure_git_user(path)
            
            # Create .gitignore
            gitignore_path = path / '.gitignore'
            if not gitignore_path.exists():
                with open(gitignore_path, 'w') as f:
                    f.write(GITIGNORE_TEMPLATE)
                self.logger.info(f"Created .gitignore in {path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize git repo in {path}: {e}")
            return False
    
    def check_status(self, path: Path) -> Tuple[bool, str]:
        """Check git status. Returns (is_clean, status_output)."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=path,
                capture_output=True,
                text=True,
                check=True
            )
            is_clean = len(result.stdout.strip()) == 0
            return is_clean, result.stdout
        except Exception as e:
            self.logger.error(f"Failed to check git status in {path}: {e}")
            return False, ""
    
    def commit_changes(self, path: Path, message: str) -> bool:
        """Add all changes and commit with message."""
        try:
            # Add all files
            subprocess.run(['git', 'add', '-A'], cwd=path, capture_output=True, check=True)
            
            # Commit
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=path,
                capture_output=True,
                check=True
            )
            self.logger.info(f"Committed changes in {path}: {message}")
            return True
        except subprocess.CalledProcessError as e:
            # git commit returns non-zero if there's nothing to commit
            if b'nothing to commit' in e.stdout or b'nothing to commit' in e.stderr:
                self.logger.info(f"No changes to commit in {path}")
                return True
            self.logger.error(f"Failed to commit changes in {path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to commit changes in {path}: {e}")
            return False
    
    def manage_directory(self, path: Path, operation: str) -> bool:
        """Manage git for a directory before and after changes."""
        if not self.is_git_installed():
            self.logger.warning("Git is not installed, skipping version control")
            return True
        
        if not self.has_git_repo(path):
            self.logger.info(f"No git repo found in {path}, initializing...")
            if not self.init_repo(path):
                return False
            # Commit initial state
            self.commit_changes(path, "Initial commit before file organizer changes")
        else:
            # Ensure git user is configured (for existing repos)
            self.configure_git_user(path)
            
            # Check if there are uncommitted changes
            is_clean, status = self.check_status(path)
            if not is_clean:
                self.logger.info(f"Uncommitted changes found in {path}, committing...")
                self.commit_changes(path, "Auto-commit before file organizer changes")
        
        return True
    
    def commit_after_sync(self, path: Path, operation: str) -> bool:
        """Commit changes after synchronization operation."""
        return self.commit_changes(path, f"File organizer: {operation}")


class FolderSynchronizer:
    """Synchronizes folders and removes duplicates."""
    
    def __init__(self, logger: logging.Logger, git_manager: GitManager):
        self.logger = logger
        self.git_manager = git_manager
    
    def _get_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Get hash of file content."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    def find_duplicates_in_directories(self, directories: List[Path]) -> Dict[str, List[Path]]:
        """Find duplicate files across multiple directories."""
        file_hashes = defaultdict(list)
        
        for directory in directories:
            if not directory.exists():
                self.logger.warning(f"Directory {directory} does not exist")
                continue
            
            try:
                for file_path in directory.rglob('*'):
                    if file_path.is_file() and not file_path.is_symlink():
                        try:
                            file_hash = self._get_file_hash(file_path)
                            if file_hash:
                                file_hashes[file_hash].append(file_path)
                        except Exception as e:
                            self.logger.warning(f"Could not process {file_path}: {e}")
            except Exception as e:
                self.logger.error(f"Error scanning directory {directory}: {e}")
        
        # Return only files with duplicates
        duplicates = {hash_val: paths for hash_val, paths in file_hashes.items() if len(paths) > 1}
        return duplicates
    
    def remove_duplicates(self, duplicates: Dict[str, List[Path]], keep_newest: bool = True) -> int:
        """Remove duplicate files, keeping the newest (or oldest)."""
        removed_count = 0
        
        for file_hash, file_paths in duplicates.items():
            if len(file_paths) <= 1:
                continue
            
            # Sort by modification time
            sorted_paths = sorted(file_paths, key=lambda p: p.stat().st_mtime, reverse=keep_newest)
            keep_file = sorted_paths[0]
            
            self.logger.info(f"Keeping: {keep_file}")
            
            for duplicate_file in sorted_paths[1:]:
                try:
                    duplicate_file.unlink()
                    self.logger.info(f"Removed duplicate: {duplicate_file}")
                    removed_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to remove {duplicate_file}: {e}")
        
        return removed_count
    
    def sync_directories(self, source: Path, target: Path, sync_mode: str = 'newer') -> bool:
        """
        Synchronize two directories.
        sync_mode: 'newer' (newer files override), 'source' (source overrides all)
        """
        if not source.exists():
            self.logger.error(f"Source directory {source} does not exist")
            return False
        
        target.mkdir(parents=True, exist_ok=True)
        
        # Manage git before changes
        self.git_manager.manage_directory(target, f"sync from {source}")
        
        try:
            for source_item in source.rglob('*'):
                if source_item.is_file():
                    relative_path = source_item.relative_to(source)
                    target_item = target / relative_path
                    
                    # Create parent directories
                    target_item.parent.mkdir(parents=True, exist_ok=True)
                    
                    should_copy = False
                    
                    if not target_item.exists():
                        should_copy = True
                    elif sync_mode == 'source':
                        should_copy = True
                    elif sync_mode == 'newer':
                        source_mtime = source_item.stat().st_mtime
                        target_mtime = target_item.stat().st_mtime
                        if source_mtime > target_mtime:
                            should_copy = True
                    
                    if should_copy:
                        try:
                            if source_item.is_symlink():
                                # Preserve symlinks
                                link_target = os.readlink(source_item)
                                if target_item.exists() or target_item.is_symlink():
                                    target_item.unlink()
                                target_item.symlink_to(link_target)
                                self.logger.info(f"Copied symlink: {source_item} -> {target_item}")
                            else:
                                shutil.copy2(source_item, target_item)
                                self.logger.info(f"Copied: {source_item} -> {target_item}")
                        except Exception as e:
                            self.logger.error(f"Failed to copy {source_item} to {target_item}: {e}")
            
            # Commit changes
            self.git_manager.commit_after_sync(target, f"Synchronized from {source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during synchronization: {e}")
            return False


class BackgroundBackup:
    """Handles background backup to a remote/backup drive."""
    
    def __init__(self, logger: logging.Logger, config: Dict):
        self.logger = logger
        self.config = config
        self.running = False
        self.backup_queue = queue.Queue()
        self.backup_drive_path = Path(config.get('backup_drive_path', ''))
        if not self.backup_drive_path or str(self.backup_drive_path) == '':
            self.logger.warning("Backup drive path not configured - backup disabled")
        self.notification_sent = False
    
    def is_backup_drive_online(self) -> bool:
        """Check if backup drive is accessible."""
        try:
            return self.backup_drive_path.exists() and os.access(self.backup_drive_path, os.W_OK)
        except Exception as e:
            self.logger.warning(f"Backup drive check failed: {e}")
            return False
    
    def send_notification(self, message: str):
        """Send notification to user (macOS notification)."""
        try:
            subprocess.run([
                'osascript', '-e',
                f'display notification "{message}" with title "File Organizer"'
            ])
            self.logger.info(f"Notification sent: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def copy_with_symlink_preservation(self, source: Path, target: Path):
        """Copy file or directory preserving symlinks."""
        if source.is_symlink():
            # Get the link target
            link_target = os.readlink(source)
            
            # If it's an absolute path, convert it to the backup drive equivalent
            if os.path.isabs(link_target):
                link_path = Path(link_target)
                # Try to make it relative to a known base path
                for base_path in [Path.home(), Path.cwd()]:
                    try:
                        rel_path = link_path.relative_to(base_path)
                        # Map to backup drive
                        new_link_target = self.backup_drive_path / rel_path
                        link_target = str(new_link_target)
                        break
                    except ValueError:
                        continue
            
            # Create the symlink
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(link_target)
            self.logger.info(f"Copied symlink: {source} -> {target} (points to {link_target})")
        else:
            # Regular file copy
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            self.logger.info(f"Copied file: {source} -> {target}")
    
    def backup_item(self, source_path: Path):
        """Backup a single file or directory to backup drive."""
        if not self.is_backup_drive_online():
            if not self.notification_sent:
                self.send_notification("Backup drive is offline. Backup paused.")
                self.notification_sent = True
            return False
        
        self.notification_sent = False
        
        try:
            # Calculate target path in backup drive
            # Try to maintain directory structure relative to home or current directory
            try:
                rel_path = source_path.relative_to(Path.home())
                target_path = self.backup_drive_path / rel_path
            except ValueError:
                try:
                    rel_path = source_path.relative_to(Path.cwd())
                    target_path = self.backup_drive_path / rel_path
                except ValueError:
                    # If not under home or cwd, use a backup folder
                    target_path = self.backup_drive_path / 'backup' / source_path.name
            
            if source_path.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self.copy_with_symlink_preservation(source_path, target_path)
            elif source_path.is_dir():
                for item in source_path.rglob('*'):
                    if item.is_file():
                        rel_item = item.relative_to(source_path)
                        target_item = target_path / rel_item
                        target_item.parent.mkdir(parents=True, exist_ok=True)
                        self.copy_with_symlink_preservation(item, target_item)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup {source_path}: {e}")
            return False
    
    def backup_worker(self):
        """Background worker thread for backups."""
        self.logger.info("Backup worker started")
        
        while self.running:
            try:
                # Check if backup drive is online
                if not self.is_backup_drive_online():
                    if not self.notification_sent:
                        self.send_notification("Backup drive is offline. Waiting...")
                        self.notification_sent = True
                    time.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Get item from queue with timeout
                try:
                    item = self.backup_queue.get(timeout=10)
                    self.backup_item(item)
                    self.backup_queue.task_done()
                    
                    # Sleep to keep CPU usage low
                    time.sleep(1)
                except queue.Empty:
                    pass
                
            except Exception as e:
                self.logger.error(f"Error in backup worker: {e}")
                time.sleep(60)
        
        self.logger.info("Backup worker stopped")
    
    def start(self):
        """Start the backup worker thread."""
        self.running = True
        self.backup_thread = threading.Thread(target=self.backup_worker, daemon=True)
        self.backup_thread.start()
    
    def stop(self):
        """Stop the backup worker thread."""
        self.running = False
        if hasattr(self, 'backup_thread'):
            self.backup_thread.join(timeout=10)
    
    def queue_backup(self, path: Path):
        """Add a path to the backup queue."""
        self.backup_queue.put(path)


class EnhancedFileOrganizer:
    """Enhanced file organizer with full production features."""
    
    def __init__(self, config_file: str = "organizer_config.json", test_mode: bool = False):
        self.config_file = config_file
        self.test_mode = test_mode
        self.config = self._load_config()
        self.running = True  # Always set to True initially (even for --scan-once)
        self.logger = self._setup_logging()
        
        # Validate and resolve drive placeholders (skip in test mode)
        if not test_mode:
            self._validate_and_resolve_drives()
        
        # Initialize managers
        self.git_manager = GitManager(self.logger, self.config)
        self.folder_sync = FolderSynchronizer(self.logger, self.git_manager)
        self.background_backup = BackgroundBackup(self.logger, self.config)
        self.content_analyzer = DynamicContentAnalyzer(self.logger, self.config)
        
        # File classification rules
        self.file_type_mappings = {
            'documents': ['.txt', '.doc', '.docx', '.pdf', '.rtf', '.odt'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
            'videos': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php'],
        }
        
        # Track processed files
        self.processed_files = set()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"ERROR: Could not load config file {self.config_file}: {e}")
                sys.exit(1)
        else:
            print("\n" + "=" * 70)
            print("ERROR: Configuration file not found!")
            print("=" * 70)
            print(f"\nThe file '{self.config_file}' does not exist.")
            
            # Check if template exists
            template_file = "organizer_config.template.json"
            if os.path.exists(template_file):
                print("\nTo get started:")
                print(f"  1. Copy the template:  cp {template_file} {self.config_file}")
                print(f"  2. Edit your config:   nano {self.config_file}")
                print("  3. Update the 'drives' section with your actual paths")
                print(f"  4. Run the program again")
            else:
                print(f"\nPlease create {self.config_file} with your drive configurations.")
            
            print("\n" + "=" * 70 + "\n")
            sys.exit(1)
    
    def _validate_and_resolve_drives(self):
        """Validate drive configuration and resolve placeholders."""
        # Check if drives are configured
        if 'drives' not in self.config:
            print("ERROR: 'drives' section missing from config file!")
            print("Please add a 'drives' section to your config file.")
            sys.exit(1)
        
        drives = self.config['drives']
        unconfigured = []
        
        # Check for unconfigured drives
        for drive_name, drive_path in drives.items():
            if isinstance(drive_path, str) and ('UNCONFIGURED' in drive_path or drive_path.strip() == ''):
                unconfigured.append(drive_name)
        
        if unconfigured:
            print("\n" + "=" * 70)
            print("ERROR: Drive configuration required!")
            print("=" * 70)
            print(f"\nThe following drives are not configured in {self.config_file}:")
            for drive in unconfigured:
                print(f"  - {drive}: {drives[drive]}")
            print("\nPlease edit the config file and replace the placeholder paths")
            print("with your actual drive paths.")
            print("\nExample:")
            print('  "MAIN_DRIVE": "/Users/yourname"')
            print('  "GOOGLE_DRIVE": "/Users/yourname/Google Drive"')
            print('  "BACKUP_DRIVE": "/Volumes/BackupDrive"')
            print('  "EXTERNAL_DRIVE": "/Volumes/YourDrive"')
            print("\n" + "=" * 70)
            sys.exit(1)
        
        # Resolve drive placeholders in config
        self._resolve_placeholders('source_folders')
        self._resolve_placeholders('exclude_folders')
        self._resolve_placeholder_value('output_base')
        self._resolve_placeholder_value('backup_drive_path')
        
        # Resolve in sync_pairs
        if 'sync_pairs' in self.config:
            for pair in self.config['sync_pairs']:
                if 'source' in pair:
                    pair['source'] = self._resolve_path(pair['source'])
                if 'target' in pair:
                    pair['target'] = self._resolve_path(pair['target'])
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path containing drive placeholders."""
        drives = self.config['drives']
        for drive_name, drive_path in drives.items():
            if path.startswith(drive_name + '/') or path.startswith(drive_name):
                return path.replace(drive_name, drive_path, 1)
        return path
    
    def _resolve_placeholders(self, config_key: str):
        """Resolve drive placeholders in a list of paths."""
        if config_key in self.config:
            self.config[config_key] = [
                self._resolve_path(path) for path in self.config[config_key]
            ]
    
    def _resolve_placeholder_value(self, config_key: str):
        """Resolve drive placeholder in a single path value."""
        if config_key in self.config:
            self.config[config_key] = self._resolve_path(self.config[config_key])
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('file_organizer_v2')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = Path.home() / '.file_organizer.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _safe_path_operation(self, operation, *args, **kwargs):
        """Safely execute path operations with retry logic for flaky volumes."""
        for attempt in range(self.config['flaky_volume_retries']):
            try:
                return operation(*args, **kwargs)
            except (OSError, IOError, PermissionError) as e:
                if attempt < self.config['flaky_volume_retries'] - 1:
                    self.logger.warning(f"Path operation failed (attempt {attempt + 1}): {e}")
                    time.sleep(self.config['retry_delay'])
                else:
                    self.logger.error(f"Path operation failed after {self.config['flaky_volume_retries']} attempts: {e}")
                    raise
    
    def should_exclude_path(self, path: Path) -> bool:
        """Check if path should be excluded from processing."""
        path_str = str(path)
        for exclude in self.config['exclude_folders']:
            if path_str.startswith(exclude):
                return True
        return False
    
    def classify_by_type(self, file_path: Path) -> List[str]:
        """Classify file by its type/extension."""
        classifications = []
        extension = file_path.suffix.lower()
        
        for category, extensions in self.file_type_mappings.items():
            if extension in extensions:
                classifications.append(category)
                
        return classifications
    
    def classify_by_year(self, file_path: Path) -> List[str]:
        """
        Classify file by year.
        Priority: 1) Year in filename, 2) ctime, 3) mtime, 4) atime
        Valid year range: 1752-2035 (Gregorian calendar adoption to near future)
        """
        years = []
        MIN_YEAR = 1752  # Gregorian calendar adoption in Britain
        MAX_YEAR = 2035  # Reasonable future limit
        
        # Priority 1: Extract year from filename (YYYYMMDD or YYYY-MM-DD patterns)
        filename = file_path.stem  # Without extension
        
        # Pattern 1: YYYYMMDD at start (e.g., 20240101-fishing-trip.txt)
        match = re.match(r'^(\d{4})\d{2}\d{2}', filename)
        if match:
            year = match.group(1)
            year_int = int(year)
            if MIN_YEAR <= year_int <= MAX_YEAR:
                years.append(year)
                self.logger.debug(f"Found year {year} in filename: {file_path.name}")
                return years
            else:
                self.logger.debug(f"Rejected year {year} (out of range {MIN_YEAR}-{MAX_YEAR}): {file_path.name}")
        
        # Pattern 2: YYYY-MM-DD anywhere (e.g., backup-2024-01-01.txt)
        match = re.search(r'(\d{4})-\d{2}-\d{2}', filename)
        if match:
            year = match.group(1)
            year_int = int(year)
            if MIN_YEAR <= year_int <= MAX_YEAR:
                years.append(year)
                self.logger.debug(f"Found year {year} in filename: {file_path.name}")
                return years
            else:
                self.logger.debug(f"Rejected year {year} (out of range): {file_path.name}")
        
        # Pattern 3: YYYY anywhere in filename (e.g., report2024.doc)
        matches = re.findall(r'\b(\d{4})\b', filename)
        valid_matches = [y for y in matches if MIN_YEAR <= int(y) <= MAX_YEAR]
        if valid_matches:
            # If multiple valid years found, use the earliest one
            year = min(valid_matches)
            years.append(year)
            self.logger.debug(f"Found year {year} in filename: {file_path.name}")
            return years
        
        # Priority 2-4: Fall back to file metadata (ctime > mtime > atime)
        try:
            stat = file_path.stat()
            
            # Try ctime (creation time on Unix, or metadata change time)
            if hasattr(stat, 'st_birthtime'):
                # macOS has birthtime (true creation time)
                year = datetime.fromtimestamp(stat.st_birthtime).year
                years.append(str(year))
                self.logger.debug(f"Using birthtime year {year} for: {file_path.name}")
            elif hasattr(stat, 'st_ctime'):
                # Fall back to ctime
                year = datetime.fromtimestamp(stat.st_ctime).year
                years.append(str(year))
                self.logger.debug(f"Using ctime year {year} for: {file_path.name}")
            elif hasattr(stat, 'st_mtime'):
                # Fall back to mtime
                year = datetime.fromtimestamp(stat.st_mtime).year
                years.append(str(year))
                self.logger.debug(f"Using mtime year {year} for: {file_path.name}")
            elif hasattr(stat, 'st_atime'):
                # Last resort: atime
                year = datetime.fromtimestamp(stat.st_atime).year
                years.append(str(year))
                self.logger.debug(f"Using atime year {year} for: {file_path.name}")
                
        except Exception as e:
            self.logger.debug(f"Could not get year from metadata for {file_path}: {e}")
        
        return years
    
    def extract_text_from_video(self, file_path: Path) -> str:
        """Extract text from video by OCR on frames."""
        if not cv2 or not pytesseract:
            return ""
        
        try:
            # Open video
            video = cv2.VideoCapture(str(file_path))
            if not video.isOpened():
                return ""
            
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or total_frames == 0:
                return ""
            
            # Sample frames (every 10 seconds, max 20 frames)
            frame_interval = int(fps * 10)  # 10 seconds
            max_frames = 20
            
            extracted_text = []
            frame_count = 0
            sampled = 0
            
            while sampled < max_frames:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Process frame at intervals
                if frame_count % frame_interval == 0:
                    # Convert to PIL Image for OCR
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Extract text from frame
                    text = pytesseract.image_to_string(pil_image)
                    if text.strip():
                        extracted_text.append(text.strip())
                    
                    sampled += 1
                
                frame_count += 1
            
            video.release()
            return '\n'.join(extracted_text)
            
        except Exception as e:
            self.logger.debug(f"Could not extract text from video {file_path}: {e}")
            return ""
    
    def extract_text_from_scanned_pdf(self, file_path: Path) -> str:
        """Extract text from scanned PDF using OCR."""
        if not convert_from_path or not pytesseract:
            return ""
        
        try:
            # Convert PDF pages to images (limit to first 10 pages)
            images = convert_from_path(str(file_path), first_page=1, last_page=10)
            
            extracted_text = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                if text.strip():
                    extracted_text.append(text.strip())
            
            return '\n'.join(extracted_text)
            
        except Exception as e:
            self.logger.debug(f"Could not OCR PDF {file_path}: {e}")
            return ""
    
    def extract_text_content(self, file_path: Path) -> str:
        """Extract text content from various file types."""
        try:
            extension = file_path.suffix.lower()
            
            # Limit file size for content extraction
            if file_path.stat().st_size > self.config.get('max_file_size', 100 * 1024 * 1024):
                return ""
            
            # Plain text files
            if extension == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            # Microsoft Word .docx (Office Open XML)
            elif extension == '.docx' and docx:
                doc = docx.Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            # Rich Text Format
            elif extension == '.rtf' and rtf_to_text:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    rtf_content = f.read()
                    return rtf_to_text(rtf_content)
            
            # OpenDocument Text (.odt)
            elif extension == '.odt' and odf_load and odf_text:
                doc = odf_load(str(file_path))
                paragraphs = doc.getElementsByType(odf_text.P)
                return '\n'.join([str(p) for p in paragraphs])
            
            # Old Microsoft Word .doc
            elif extension == '.doc':
                # Try antiword first (best for .doc files)
                try:
                    result = subprocess.run(
                        ['antiword', str(file_path)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout
                except Exception as e:
                    self.logger.debug(f"antiword failed for {file_path}: {e}")
                
                # Fallback: try to extract readable text from binary
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        text = content.decode('latin-1', errors='ignore')
                        # Filter out binary garbage, keep only readable text
                        readable = ''.join(c for c in text if c.isprintable() or c.isspace())
                        # Only return if we got something reasonable
                        if len(readable.strip()) > 50:
                            return readable
                except:
                    pass
            
            # PDF files
            elif extension == '.pdf' and PyPDF2:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages[:10]:  # Limit to first 10 pages
                        text += page.extract_text()
                    
                    # If no text found, try OCR (scanned PDF)
                    if not text.strip() and convert_from_path and pytesseract:
                        self.logger.debug(f"No text in PDF, trying OCR: {file_path}")
                        text = self.extract_text_from_scanned_pdf(file_path)
                    
                    return text
            
            # OCR for images
            elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                self.logger.info(f"Processing image: {file_path.name}")
                image = Image.open(file_path)
                
                all_text = []
                
                # Method 1: Try EasyOCR first (better with artistic text)
                easyocr_reader = self.content_analyzer.get_easyocr_reader()
                if easyocr_reader:
                    try:
                        import numpy as np
                        img_array = np.array(image)
                        ocr_results = easyocr_reader.readtext(img_array)
                        # Extract text from results (format: [(bbox, text, confidence)])
                        easyocr_text = ' '.join([text for (bbox, text, conf) in ocr_results if conf > 0.3])
                        if easyocr_text.strip():
                            all_text.append(easyocr_text)
                            self.logger.info(f"EasyOCR found: {easyocr_text[:50]}...")
                    except Exception as e:
                        self.logger.debug(f"EasyOCR failed: {e}")
                
                # Method 2: CLIP for object/scene recognition (optional - slow on CPU)
                use_clip = self.config.get('ml_content_analysis', {}).get('use_clip', False)
                if use_clip:
                    clip_model, clip_processor = self.content_analyzer.get_clip_model()
                if use_clip and clip_model and clip_processor:
                    try:
                        # Define categories to check
                        candidate_labels = [
                            "fish", "fishing", "ocean", "sea", "water",
                            "person", "portrait", "woman", "man",
                            "music", "concert", "performance", "singer",
                            "document", "text", "writing",
                            "nature", "animal", "food", "building"
                        ]
                        
                        # Process image with CLIP
                        inputs = clip_processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
                        outputs = clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                        
                        # Get top 3 matches above confidence threshold
                        top_indices = probs[0].argsort(descending=True)[:3]
                        clip_keywords = []
                        for idx in top_indices:
                            confidence = probs[0][idx].item()
                            if confidence > 0.15:  # 15% confidence threshold
                                label = candidate_labels[idx]
                                clip_keywords.append(label)
                                self.logger.info(f"CLIP detected '{label}' ({confidence:.2f}) in {file_path.name}")
                        
                        if clip_keywords:
                            all_text.append(' '.join(clip_keywords))
                    except Exception as e:
                        self.logger.debug(f"CLIP failed: {e}")
                
                # Method 3: Tesseract OCR (fallback)
                if pytesseract:
                    try:
                        # Try different PSM modes
                        for psm in [3, 6, 11, 12]:
                            try:
                                config = f'--psm {psm}'
                                text = pytesseract.image_to_string(image, config=config)
                                if text.strip():
                                    all_text.append(text)
                            except:
                                pass
                    except:
                        pass
                
                # Combine all extracted text
                combined_text = '\n'.join(all_text)
                if combined_text.strip():
                    self.logger.info(f"Total extracted {len(combined_text)} chars from {file_path.name}")
                    return combined_text
                
                self.logger.info(f"No text/objects found in {file_path.name}")
                return ""
            
            # Video files - extract text from frames
            elif extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'] and cv2 and pytesseract:
                self.logger.info(f"Extracting text from video: {file_path.name}")
                return self.extract_text_from_video(file_path)
                
        except Exception as e:
            self.logger.debug(f"Could not extract text from {file_path}: {e}")
            
        return ""
    
    def create_soft_link(self, source_path: Path, target_folder: str, target_name: str) -> bool:
        """Create a soft link from source to target folder."""
        try:
            target_dir = Path(self.config['output_base']) / target_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / target_name
            
            # Check if link already exists and points to correct location
            if target_path.is_symlink():
                try:
                    existing_target = os.readlink(target_path)
                    relative_source = os.path.relpath(source_path, target_dir)
                    if existing_target == relative_source:
                        # Link already exists and is correct - skip
                        return True
                except:
                    pass
                # Link exists but is wrong - remove it
                target_path.unlink()
            elif target_path.exists():
                # Regular file exists - remove it
                target_path.unlink()
            
            # Create relative symlink
            relative_source = os.path.relpath(source_path, target_dir)
            target_path.symlink_to(relative_source)
            
            self.logger.debug(f"Created soft link: {target_path} -> {relative_source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create soft link {source_path} -> {target_folder}/{target_name}: {e}")
            return False
    
    def process_file(self, file_path: Path) -> None:
        """Process a single file with dynamic content analysis."""
        if str(file_path) in self.processed_files:
            return
        
        if self.should_exclude_path(file_path):
            return
        
        try:
            # Extract content for analysis
            content = ""
            if self.config.get('enable_content_analysis', True):
                content = self.extract_text_content(file_path)
            
            # Analyze file with dynamic content analyzer
            # This learns keywords from both path and content
            self.content_analyzer.analyze_file(file_path, content)
            
            # Classify file by type and year
            classifications = set()
            classifications.update(self.classify_by_type(file_path))
            classifications.update(self.classify_by_year(file_path))
            
            # Create soft links for basic classifications
            for classification in classifications:
                self.create_soft_link(file_path, classification, file_path.name)
            
            self.processed_files.add(str(file_path))
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
    
    def scan_directory(self, directory: Path, max_depth: int = 10) -> None:
        """Scan directory and process files."""
        if not directory.exists() or self.should_exclude_path(directory):
            return
        
        # Check if we should stop
        if hasattr(self, 'running') and not self.running:
            return
        
        self.logger.info(f"Scanning directory: {directory}")
        
        try:
            for item in directory.iterdir():
                # Check if we should stop
                if hasattr(self, 'running') and not self.running:
                    return
                    
                if item.is_file() and not item.is_symlink():
                    self.process_file(item)
                elif item.is_dir() and max_depth > 0:
                    self.scan_directory(item, max_depth - 1)
        except PermissionError:
            self.logger.warning(f"Permission denied: {directory}")
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory}: {e}")
    
    def sync_configured_folders(self) -> None:
        """Synchronize configured folder pairs."""
        if not self.config['enable_folder_sync']:
            return
        
        self.logger.info("Starting folder synchronization")
        
        for sync_pair in self.config.get('sync_pairs', []):
            source = Path(sync_pair['source'])
            target = Path(sync_pair['target'])
            
            if source.exists():
                self.logger.info(f"Syncing {source} -> {target}")
                self.folder_sync.sync_directories(source, target, sync_mode='newer')
            else:
                self.logger.warning(f"Source folder does not exist: {source}")
    
    def remove_duplicates_across_system(self) -> None:
        """Find and remove duplicates across all source folders."""
        if not self.config['enable_duplicate_detection']:
            return
        
        self.logger.info("Scanning for duplicate files")
        
        directories = [Path(folder) for folder in self.config['source_folders']]
        duplicates = self.folder_sync.find_duplicates_in_directories(directories)
        
        if duplicates:
            self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
            removed = self.folder_sync.remove_duplicates(duplicates, keep_newest=True)
            self.logger.info(f"Removed {removed} duplicate files")
    
    def queue_background_backup(self) -> None:
        """Queue important directories for background backup."""
        if not self.config.get('enable_background_backup', False):
            return
        
        # Get backup directories from config
        backup_dirs = self.config.get('backup_directories', [])
        
        if not backup_dirs:
            self.logger.info("No backup directories configured - skipping background backup")
            return
        
        self.logger.info("Queueing items for background backup")
        
        for dir_path in backup_dirs:
            directory = Path(dir_path)
            if directory.exists():
                self.background_backup.queue_backup(directory)
            else:
                self.logger.warning(f"Backup directory does not exist: {dir_path}")
    
    def run_full_cycle(self) -> None:
        """Run a complete organization cycle with dynamic content discovery."""
        self.logger.info("=" * 80)
        self.logger.info("Starting full organization cycle")
        self.logger.info("=" * 80)
        
        # Step 1: Sync configured folders
        if hasattr(self, 'running') and not self.running:
            return
        self.sync_configured_folders()
        
        # Step 2: Remove duplicates
        if hasattr(self, 'running') and not self.running:
            return
        self.remove_duplicates_across_system()
        
        # Step 3: Scan and organize files (this builds keyword frequencies)
        for folder in self.config['source_folders']:
            if hasattr(self, 'running') and not self.running:
                return
            folder_path = Path(folder)
            if folder_path.exists():
                self._safe_path_operation(self.scan_directory, folder_path)
        
        # Step 4: Discover content categories based on learned keywords
        if hasattr(self, 'running') and not self.running:
            return
        if self.config.get('ml_content_analysis', {}).get('enabled', True):
            self.logger.info("Discovering content categories from analyzed files...")
            discovered_categories = self.content_analyzer.discover_categories()
            
            # Create soft links for discovered categories
            for category_name, file_paths in discovered_categories.items():
                if hasattr(self, 'running') and not self.running:
                    return
                for file_path_str in file_paths:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        self.create_soft_link(file_path, category_name, file_path.name)
            
            # Save discovered categories for review
            categories_file = Path.home() / '.file_organizer_discovered_categories.json'
            self.content_analyzer.save_discovered_categories(categories_file)
            self.logger.info(f"Discovered {len(discovered_categories)} content categories")
        
        # Step 5: Queue background backup
        if hasattr(self, 'running') and not self.running:
            return
        self.queue_background_backup()
        
        self.logger.info("Full organization cycle complete")
    
    def run_daemon(self) -> None:
        """Run the organizer as a background daemon."""
        self.running = True
        self.logger.info("Enhanced File Organizer daemon started")
        print("\n" + "=" * 70)
        print("File Organizer is running. Press Ctrl-C to stop.")
        print("=" * 70 + "\n")
        
        # Start background backup worker
        if self.config.get('enable_background_backup', False):
            self.background_backup.start()
        
        # Track Ctrl-C presses for force quit
        self.interrupt_count = 0
        
        def signal_handler(signum, frame):
            self.interrupt_count += 1
            
            if self.interrupt_count == 1:
                print("\n\n" + "=" * 70)
                print("Stopping File Organizer... (this may take a moment)")
                print("Press Ctrl-C again to force quit immediately")
                print("=" * 70 + "\n")
                self.logger.info("Received shutdown signal - stopping gracefully")
                self.running = False
            else:
                print("\n\n" + "=" * 70)
                print("Force quitting...")
                print("=" * 70 + "\n")
                self.logger.warning("Force quit - some operations may not have completed")
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while self.running:
            try:
                self.run_full_cycle()
                if not self.running:
                    break
                    
                self.logger.info(f"Sleeping for {self.config['scan_interval']} seconds (Ctrl-C to stop)")
                
                # Sleep in smaller chunks so Ctrl-C is more responsive
                sleep_remaining = self.config['scan_interval']
                while sleep_remaining > 0 and self.running:
                    time.sleep(min(1, sleep_remaining))
                    sleep_remaining -= 1
                    
            except KeyboardInterrupt:
                print("\n\nStopping...")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in daemon loop: {e}")
                if self.running:
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        # Stop background backup worker
        if self.config.get('enable_background_backup', False):
            print("Stopping background backup worker...")
            self.background_backup.stop()
        
        print("\n" + "=" * 70)
        print("File Organizer stopped successfully")
        print("=" * 70 + "\n")
        self.logger.info("Enhanced File Organizer daemon stopped")


def create_test_environment():
    """Create test folders and files as specified in requirements."""
    # Create test directory
    test_dir = Path('test')
    test_dir.mkdir(exist_ok=True)
    
    # Examples folder for reference files (images, docs, etc.)
    examples_dir = Path('examples')
    
    test_files = {
        'foo': [
            '20240101-fishing-trip.txt',
            'something-else.docx',  # Will copy from examples/ (real docx about fishing)
            'announcement.doc',  # Will copy from examples/ (real doc file)
            'has-text.jpg',  # Will copy from examples/ (Peggy Lee image with text)
            'peggy-lee-concert.txt',
            'music-collection.txt',
            'jazz-playlist.txt',
            'philosophy-notes.txt',
            '28980328.jpg'  # Will copy from examples/ (tests year 2898 rejection)
        ],
        'bar': [
            'something.png',  # Will copy from examples/ (fish photo)
            'vinyl-records.txt'
        ],
        'baz': [
            'us-webp.pdf'  # Will copy from examples/ (real PDF file)
        ]
    }
    
    print("Creating test environment in test/...")
    
    for folder, files in test_files.items():
        folder_path = test_dir / folder
        folder_path.mkdir(exist_ok=True)
        
        for file_name in files:
            file_path = folder_path / file_name
            if not file_path.exists():
                # First, check if this file exists in examples/ folder
                example_file = examples_dir / file_name
                if example_file.exists():
                    # Copy from examples
                    try:
                        import shutil
                        shutil.copy2(example_file, file_path)
                        print(f"Copied from examples: {file_path}")
                        continue
                    except Exception as e:
                        print(f"Could not copy from examples: {e}")
                        # Fall through to generate
                
                # Create files with some content
                if file_name == '20240101-fishing-trip.txt':
                    content = "Fishing trip notes from January 1st, 2024. Caught several fish."
                    with open(file_path, 'w') as f:
                        f.write(content)
                elif file_name == 'peggy-lee-concert.txt':
                    content = "Concert notes: Peggy Lee performed 'Is That All There Is?' at the jazz club. Amazing music performance of this philosophical song about life and existence. Her voice and the music were transcendent."
                    with open(file_path, 'w') as f:
                        f.write(content)
                elif file_name == 'music-collection.txt':
                    content = "My music collection includes jazz, blues, and classical music. Artists like Peggy Lee, Ella Fitzgerald, and Miles Davis. Music is essential to life."
                    with open(file_path, 'w') as f:
                        f.write(content)
                elif file_name == 'jazz-playlist.txt':
                    content = "Jazz playlist: Peggy Lee - Is That All There Is, Ella Fitzgerald - Summertime, Miles Davis - So What. Great music for evening listening."
                    with open(file_path, 'w') as f:
                        f.write(content)
                elif file_name == 'vinyl-records.txt':
                    content = "Vinyl records collection: Peggy Lee albums, jazz music from the 1960s. The sound quality of vinyl music is unmatched."
                    with open(file_path, 'w') as f:
                        f.write(content)
                elif file_name == 'philosophy-notes.txt':
                    content = "Philosophy notes: Existentialism and the question 'Is that all there is?' - exploring meaning in life. Philosophy helps us understand existence."
                    with open(file_path, 'w') as f:
                        f.write(content)
                else:
                    # Generic fallback for any other files not in examples/
                    # Create simple text placeholder
                    content = f"Test content for {file_name}"
                    with open(file_path, 'w') as f:
                        f.write(content)
                
                # Set proper creation dates for test files
                if file_name == '20240101-fishing-trip.txt':
                    # Set creation date to January 1, 2024
                    import os
                    os.utime(file_path, (1704067200, 1704067200))  # Jan 1, 2024
                
                print(f"Created {file_path}")


def get_test_config():
    """Get test mode configuration."""
    # Get the current directory (file-organiser project directory)
    import os
    current_dir = os.getcwd()
    test_dir = os.path.join(current_dir, "test")
    
    return {
        "scan_interval": 300,
        "source_folders": [test_dir],  # Scan test directory only
        "exclude_folders": [
            # Exclude the organized output folder to avoid recursion
            os.path.join(test_dir, "organized"),
        ],
        "output_base": os.path.join(test_dir, "organized"),
        "max_file_size": 104857600,
        "enable_content_analysis": True,
        "enable_duplicate_detection": False,  # Disable for test
        "enable_folder_sync": False,  # Disable for test
        "enable_git_tracking": False,  # Disable for test
        "enable_background_backup": False,  # Disable for test
        "flaky_volume_retries": 3,
        "retry_delay": 5,
        "sync_pairs": [],
        "ml_content_analysis": {
            "enabled": True,
            "min_keyword_frequency": 3,  # Increased to reduce noise
            "min_category_size": 3,  # Increased to reduce trivial categories
            "max_categories": 15,  # Reduced max to keep it focused
            "stop_words_enabled": True,
            "use_clip": False  # Disable CLIP by default (very slow on CPU, enable for GPU)
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='File Organizer using AI/ML',
        epilog='Run without arguments for TEST MODE (foo/bar/baz folders). Use --REAL for PRODUCTION MODE.'
    )
    parser.add_argument('-R', '--REAL', action='store_true',
                       help='Run in PRODUCTION MODE on your entire file system (default: TEST MODE)')
    parser.add_argument('--config', default=None,
                       help='Configuration file path (default: auto-detect based on mode)')
    parser.add_argument('--scan-once', action='store_true',
                       help='Run a single organization cycle instead of daemon mode')
    parser.add_argument('--sync-only', action='store_true',
                       help='Only synchronize configured folders (production mode only)')
    parser.add_argument('--dedupe-only', action='store_true',
                       help='Only remove duplicates (production mode only)')
    parser.add_argument('--create-test', action='store_true',
                       help='Create test environment and exit')
    
    args = parser.parse_args()
    
    # Handle test environment creation
    if args.create_test:
        create_test_environment()
        return
    
    # Determine mode
    production_mode = args.REAL
    
    if production_mode:
        print("=" * 70)
        print("PRODUCTION MODE - Operating on your entire file system")
        print("=" * 70)
        config_file = args.config or 'organizer_config.json'
    else:
        print("=" * 70)
        print("TEST MODE - Operating on test/ directory")
        print("Run with --REAL or -R for production mode")
        print("=" * 70)
        
        # Create test environment if it doesn't exist
        test_dir = Path('test')
        if not test_dir.exists() or not (test_dir / 'foo').exists():
            print("\nTest directory not found. Creating test environment...")
            create_test_environment()
            print()
        
        # Use test config
        config_file = 'test_config_temp.json'
        import json
        with open(config_file, 'w') as f:
            json.dump(get_test_config(), f, indent=2)
    
    # Create organizer
    organizer = EnhancedFileOrganizer(config_file, test_mode=not production_mode)
    
    # Execute based on options
    if production_mode and args.sync_only:
        organizer.sync_configured_folders()
    elif production_mode and args.dedupe_only:
        organizer.remove_duplicates_across_system()
    elif args.scan_once:
        # Set up signal handler for scan-once mode too
        def signal_handler(signum, frame):
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        organizer.run_full_cycle()
    else:
        organizer.run_daemon()


if __name__ == '__main__':
    main()

