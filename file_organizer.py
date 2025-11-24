#!/usr/bin/env python
"""
File Organizer using AI/ML

AI-Generated Code: Created using Cursor AI/ML with human guidance and testing.
Author: Rod (https://github.com/pdxrod)
License: MIT (see LICENSE file)

WARNING: This tool moves, copies, syncs, and can DELETE files. Use at your own risk.
Always test in TEST MODE first and maintain current backups.

Unified version with test and production modes:
- DEFAULT (no args): Runs in TEST MODE with foo/bar/baz folders
- With --REAL or -R: Runs in PRODUCTION MODE on your entire file system

Features:
- Dynamic content discovery (learns categories from YOUR files)
- Multi-volume support across file systems
- Folder synchronization with duplicate removal (rsync with --delete)
- Git version control integration (with proper user configuration)
- Background backup to remote drives
- Soft link preservation
- Notification system for offline volumes
- Robust error handling for flaky volumes
- Advanced OCR and AI vision (EasyOCR, CLIP)
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
import concurrent.futures

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
        self.min_word_length = self.ml_config.get('min_word_length', 5)  # Configurable minimum word length
        
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
        Extracts words separated by spaces, dashes, underscores, dots.
        Only alphabetic words 4+ letters.
        """
        if not text:
            return set()
        
        # Convert to lowercase
        text = text.lower()
        
        # Split on common delimiters: spaces, dashes, underscores, dots, commas
        # This handles filenames like "31-05-18-vienna-shoppe-kirche.jpg"
        text = re.sub(r'[-_.,/\\()[\]{}]', ' ', text)
        
        # Split into words BEFORE removing numbers (to filter out words containing digits)
        raw_words = text.split()
        
        # Filter: keep only words that are purely alphabetic (reject "ZVOZ31", "1exieieuea")
        alphabetic_words = [w for w in raw_words if w.isalpha()]
        
        keywords = set()
        for word in alphabetic_words:
            # Skip if too short, too long, or is a stop word
            # Use configurable minimum length (5 in production, 4 in test mode)
            if len(word) < self.min_word_length or len(word) > 20 or word in self.stop_words:
                continue
            
            # Skip words with repeated letters (3+ in a row) like 'meee', 'aaaa'
            if re.search(r'(.)\1{2,}', word):
                continue
            
            # Skip words that are mostly consonants (random letter combinations)
            # Good words have 30-70% vowels (not too few, not too many)
            # Count 'y' as a vowel when not at the start
            vowels = sum(1 for i, c in enumerate(word) if c in 'aeiouy' and not (c == 'y' and i == 0))
            vowel_ratio = vowels / len(word)
            if vowel_ratio < 0.3 or vowel_ratio > 0.7:
                continue
            
            keywords.add(word)
        
        # Also extract multi-word phrases (bigrams) - only from consecutive words that passed filters
        for i in range(len(alphabetic_words) - 1):
            w1, w2 = alphabetic_words[i], alphabetic_words[i + 1]
            # Both words must have individually passed all filters (be in keywords set)
            if w1 in keywords and w2 in keywords:
                phrase = f"{w1}-{w2}"  # Use dash for readability
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
        path_keywords = self.extract_keywords_from_path(file_path)
        keywords.update(path_keywords)
        
        # Keywords from content
        content_keywords = set()
        if content:
            content_keywords = self.extract_keywords_from_text(content)
            keywords.update(content_keywords)
        
        
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
            # Debug: show what keywords were found and their frequencies
            if self.keyword_frequencies:
                self.logger.info(f"Debug: Found {len(self.keyword_frequencies)} keywords with frequencies:")
                for keyword, freq in sorted(self.keyword_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]:
                    self.logger.info(f"  '{keyword}': {freq} (min: {self.min_keyword_freq})")
            else:
                self.logger.info("Debug: No keywords extracted at all")
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
            result = subprocess.run(
                ['git', 'add', '-A'], 
                cwd=path, 
                capture_output=True, 
                text=True,
                check=False  # Don't raise exception, handle errors manually
            )
            
            # Check for git errors (exit code 128 = git repo problem)
            if result.returncode != 0:
                if result.returncode == 128:
                    self.logger.warning(f"Git repository issue in {path} - skipping commit")
                    self.logger.warning(f"Git error: {result.stderr.strip()}")
                    self.logger.info("This is common in cloud storage folders (Google Drive, Dropbox)")
                    return True  # Return True to continue processing
                else:
                    self.logger.warning(f"Git add failed in {path}: {result.stderr.strip()}")
                    return True  # Still return True to not block operations
            
            # Commit
            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.logger.info(f"Committed changes in {path}: {message}")
                return True
            elif 'nothing to commit' in result.stdout or 'nothing to commit' in result.stderr:
                self.logger.info(f"No changes to commit in {path}")
                return True
            else:
                self.logger.warning(f"Git commit issue in {path}: {result.stderr.strip()}")
                return True  # Return True anyway to not block operations
                
        except Exception as e:
            self.logger.warning(f"Git operation failed in {path}: {e}")
            return True  # Return True to continue despite git issues
    
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
    
    def should_exclude_from_sync(self, path: Path, sync_excludes: List[str]) -> bool:
        """Check if path should be excluded from sync."""
        path_str = str(path)
        for exclude in sync_excludes:
            if exclude in path_str:
                return True
        return False
    
    def sync_with_rsync(self, source: Path, target: Path, sync_mode: str, sync_excludes: List[str]) -> bool:
        """Fast synchronization using rsync."""
        try:
            rsync_mode = self.git_manager.config.get('rsync_checksum_mode', 'checksum')
            self.logger.info(f"Using rsync ({rsync_mode} mode) to sync {source} -> {target}")
            
            # Build rsync command
            cmd = [
                'rsync',
                '-avh',  # archive, verbose, human-readable
                '--delete',  # Remove files in target that aren't in source
                '--stats',  # Show statistics
            ]
            
            # Add checksum flag if configured (thorough but SLOW)
            # rsync_mode already retrieved above in logger
            if rsync_mode == 'checksum':
                cmd.append('--checksum')
            # else: Use timestamp comparison (rsync default - much faster for unchanged files)

            # Optionally rely on size-only for change detection (even faster, fewer stats calls)
            if self.git_manager.config.get('rsync_size_only', False):
                cmd.append('--size-only')

            # Add additional args to reduce metadata ops on cloud/FUSE targets if configured
            # Example: --omit-dir-times --no-perms --no-group --no-owner --delete-after
            additional_args = self.git_manager.config.get('rsync_additional_args', []) or []
            for extra in additional_args:
                if isinstance(extra, str) and extra.strip():
                    cmd.append(extra.strip())
            
            # Add exclusions
            for exclude in sync_excludes:
                cmd.extend(['--exclude', exclude])
            
            # Add source and target (trailing slash important for rsync!)
            cmd.append(f'{source}/')
            cmd.append(f'{target}/')
            
            # Run rsync with timeout to prevent infinite hangs
            timeout_minutes = self.git_manager.config.get('sync_timeout_minutes', 30)
            timeout_seconds = timeout_minutes * 60
            
            self.logger.info(f"Timeout: {timeout_minutes} min | Excluding: {', '.join(sync_excludes[:5])}{'...' if len(sync_excludes) > 5 else ''}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self.logger.error(f"rsync timed out after {timeout_minutes} minutes")
                self.logger.warning(f"This sync is taking too long - skipping to prevent hanging")
                self.logger.info(f"Consider: 1) Increasing sync_timeout_minutes, 2) Reducing folder size, 3) Using chunking")
                return False
            
            # Check rsync exit code
            # 0 = success, 23 = partial transfer (some files couldn't be transferred but most worked)
            # 24 = partial transfer due to vanished source files
            if result.returncode == 0:
                # Parse rsync output for stats
                if 'Number of regular files transferred:' in result.stdout:
                    for line in result.stdout.split('\n'):
                        if 'transferred:' in line or 'speedup' in line:
                            self.logger.info(f"rsync: {line.strip()}")
                
                self.logger.info(f"rsync completed successfully")
                # No automatic git operations - git repos sync like regular folders
                return True
            elif result.returncode in [23, 24]:
                # Partial success - some files had errors but most synced OK
                self.logger.warning(f"rsync partial success (exit code {result.returncode})")
                self.logger.warning(f"Some files couldn't sync (common with FUSE mounts like ProtonDrive)")
                if 'fchmodat' in result.stderr or 'permission' in result.stderr.lower():
                    self.logger.info("This is typically harmless - files were copied but permissions couldn't be set")
                # Show first few lines of error
                error_lines = result.stderr.split('\n')[:5]
                for line in error_lines:
                    if line.strip():
                        self.logger.warning(f"  {line.strip()}")
                # No automatic git operations
                return True
            else:
                self.logger.error(f"rsync failed with exit code {result.returncode}")
                self.logger.error(f"rsync error: {result.stderr[:500]}")
                return False
                
        except Exception as e:
            self.logger.error(f"rsync sync failed: {e}")
            return False
    
    def sync_directories(self, source: Path, target: Path, sync_mode: str = 'bidirectional') -> bool:
        """
        Bidirectional synchronization between two directories.
        
        Sync logic:
        - If file in target has later date OR doesn't exist in source → copy from target to source
        - Otherwise → copy from source to target
        
        sync_mode: 'bidirectional' (default), 'newer' (newer files override), 'source' (source overrides all)
        """
        # Ensure both directories exist
        if not source.exists():
            self.logger.error(f"Source directory {source} does not exist")
            return False
        
        # Check if target drive is accessible (handle offline external drives)
        try:
            # Try to access parent volume/drive
            target_parent = target.parent
            while target_parent.parent != target_parent and not target_parent.exists():
                target_parent = target_parent.parent
            
            # If parent doesn't exist or isn't writable, the drive is offline
            if not target_parent.exists():
                self.logger.warning(f"Target drive offline or not accessible: {target}")
                self.logger.warning(f"Skipping sync - drive at {target_parent} not found")
                return False
            
            if not os.access(target_parent, os.W_OK):
                self.logger.warning(f"Target drive not writable: {target}")
                self.logger.warning(f"Skipping sync - no write access to {target_parent}")
                return False
            
            # Check disk space on target drive
            max_usage = self.git_manager.config.get('max_drive_usage_percent', 90)
            try:
                stat = os.statvfs(target_parent)
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_bfree * stat.f_frsize
                used_percent = ((total - free) / total) * 100
                
                if used_percent >= max_usage:
                    self.logger.warning(f"Target drive {target_parent} is {used_percent:.1f}% full (max: {max_usage}%)")
                    self.logger.warning(f"Skipping sync to prevent filling drive")
                    return False
                else:
                    self.logger.info(f"Target drive usage: {used_percent:.1f}% (max: {max_usage}%)")
            except Exception as e:
                self.logger.debug(f"Could not check disk space: {e}")
                
        except Exception as e:
            self.logger.warning(f"Cannot access target drive for {target}: {e}")
            return False
        
        # Create target directory
        try:
            target.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.logger.error(f"Cannot create target directory {target}: {e}")
            self.logger.warning(f"Skipping sync - target may be on offline drive")
            return False
        
        # Get sync exclusions from config
        exclude_patterns = self.git_manager.config.get('exclude_patterns', [
            'node_modules', '.git', '__pycache__', '.DS_Store', '*.pyc', 
            '.venv', 'venv', 'dist', 'build', '.next', '.cache'
        ])
        
        # For bidirectional sync, use manual Python sync (rsync is one-way)
        if sync_mode == 'bidirectional':
            return self._sync_bidirectional(source, target, exclude_patterns)
        
        # For one-way sync, try rsync first (much faster)
        use_rsync = self.git_manager.config.get('use_rsync', True)
        if use_rsync and shutil.which('rsync'):
            return self.sync_with_rsync(source, target, sync_mode, exclude_patterns)
        
        # Fallback to manual Python sync (if rsync not available)
        return self._sync_one_way(source, target, sync_mode, exclude_patterns)
    
    def _sync_bidirectional(self, source: Path, target: Path, exclude_patterns: List[str]) -> bool:
        """Bidirectional sync: copy from target to source if target is newer or missing in source, otherwise copy from source to target."""
        copied_source_to_target = 0
        copied_target_to_source = 0
        
        try:
            # Collect all files from both directories
            source_files = {}
            target_files = {}
            
            # Scan source directory
            if source.exists():
                for item in source.rglob('*'):
                    if item.is_file() and not self.should_exclude_from_sync(item, exclude_patterns):
                        relative_path = item.relative_to(source)
                        source_files[str(relative_path)] = item
            
            # Scan target directory
            if target.exists():
                for item in target.rglob('*'):
                    if item.is_file() and not self.should_exclude_from_sync(item, exclude_patterns):
                        relative_path = item.relative_to(target)
                        target_files[str(relative_path)] = item
            
            # Process all files (union of source and target)
            all_files = set(source_files.keys()) | set(target_files.keys())
            
            for relative_path_str in all_files:
                relative_path = Path(relative_path_str)
                source_file = source_files.get(relative_path_str)
                target_file = target_files.get(relative_path_str)
                
                # Determine copy direction
                copy_from_target = False
                
                if source_file is None and target_file is not None:
                    # File exists only in target → copy to source
                    copy_from_target = True
                elif source_file is not None and target_file is None:
                    # File exists only in source → copy to target
                    copy_from_target = False
                elif source_file is not None and target_file is not None:
                    # File exists in both → compare timestamps
                    try:
                        source_mtime = source_file.stat().st_mtime
                        target_mtime = target_file.stat().st_mtime
                        # If target is newer, copy from target to source
                        copy_from_target = (target_mtime > source_mtime)
                    except Exception as e:
                        self.logger.warning(f"Could not compare timestamps for {relative_path}: {e}")
                        # Default: copy from source to target
                        copy_from_target = False
                
                # Perform the copy
                if copy_from_target:
                    # Copy from target to source
                    dest_file = source / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if target_file.is_symlink():
                            link_target = os.readlink(target_file)
                            if dest_file.exists() or dest_file.is_symlink():
                                dest_file.unlink()
                            dest_file.symlink_to(link_target)
                            self.logger.debug(f"Copied symlink: {target_file} -> {dest_file}")
                        else:
                            shutil.copy2(target_file, dest_file)
                            self.logger.debug(f"Copied: {target_file} -> {dest_file}")
                        copied_target_to_source += 1
                    except Exception as e:
                        self.logger.error(f"Failed to copy {target_file} to {dest_file}: {e}")
                else:
                    # Copy from source to target
                    dest_file = target / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if source_file.is_symlink():
                            link_target = os.readlink(source_file)
                            if dest_file.exists() or dest_file.is_symlink():
                                dest_file.unlink()
                            dest_file.symlink_to(link_target)
                            self.logger.debug(f"Copied symlink: {source_file} -> {dest_file}")
                        else:
                            shutil.copy2(source_file, dest_file)
                            self.logger.debug(f"Copied: {source_file} -> {dest_file}")
                        copied_source_to_target += 1
                    except Exception as e:
                        self.logger.error(f"Failed to copy {source_file} to {dest_file}: {e}")
            
            self.logger.info(f"Bidirectional sync complete: {copied_source_to_target} files source→target, {copied_target_to_source} files target→source")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during bidirectional synchronization: {e}")
            return False
    
    def _sync_one_way(self, source: Path, target: Path, sync_mode: str, exclude_patterns: List[str]) -> bool:
        """One-way sync: copy from source to target."""
        try:
            for source_item in source.rglob('*'):
                # Skip excluded paths
                if self.should_exclude_from_sync(source_item, exclude_patterns):
                    continue
                    
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during one-way synchronization: {e}")
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
        
        # PRODUCTION MODE: Strict validation (exit on any issues)
        if not test_mode:
            self._validate_config_structure()
            self._check_unconfigured_drives()
            self._resolve_drive_placeholders()
        else:
            # TEST MODE: Always check for config file and guide user
            if not os.path.exists(self.config_file):
                print("\n" + "=" * 70)
                print("CONFIGURATION FILE REQUIRED")
                print("=" * 70)
                print(f"\nThe file '{self.config_file}' does not exist.")
                print("\nEven in test mode, you need to create a configuration file first.")
                print("This teaches you the proper setup for production mode.")
                
                # Check if template exists
                template_file = "organizer_config.template.json"
                if os.path.exists(template_file):
                    print("\nTo get started:")
                    print(f"  1. Copy the template:  cp {template_file} {self.config_file}")
                    print(f"  2. Edit your config:   nano {self.config_file}")
                    print("  3. Update the 'drives' section with your actual paths")
                    print(f"  4. Run the program again")
                    print("\nThe program will validate your config and then run in test mode.")
                else:
                    print(f"\nPlease create {self.config_file} with your drive configurations.")
                
                print("\n" + "=" * 70 + "\n")
                sys.exit(1)
            else:
                # Config file exists - validate it and use it
                print(f"Found {self.config_file} - validating configuration...")
                # Config was already loaded and validated in _load_config()
                print("Configuration validated successfully. Running in test mode with your config.")
        
        # Ensure all required config keys exist with sensible defaults
        self._ensure_config_defaults()
        
        # Set progress file path based on mode
        if test_mode:
            # In test mode, save to project directory
            self.progress_file = Path.cwd() / '.file_organizer_progress.json'
            # Override config for test mode - use test/ directory instead of user's paths
            self.config['source_folders'] = [str(Path.cwd() / 'test')]
            self.config['exclude_folders'] = [str(Path.cwd() / 'test' / 'organized')]
            self.config['output_base'] = str(Path.cwd() / 'test' / 'organized')
            self.config['enable_duplicate_detection'] = False
            self.config['enable_folder_sync'] = False
            self.config['enable_git_tracking'] = False
            self.config['enable_background_backup'] = False
            # Override ML settings for test mode to be more lenient with small dataset
            self.config['ml_content_analysis']['min_keyword_frequency'] = 2
            self.config['ml_content_analysis']['min_category_size'] = 2
            self.config['ml_content_analysis']['max_categories'] = 30
            self.config['ml_content_analysis']['min_word_length'] = 4
            # Override file size limits for test mode
            self.config['min_file_size'] = 0  # Allow tiny test files
            self.config['max_file_size'] = 10 * 1024 * 1024  # 10MB max for test files
        else:
            # In production mode, save to home directory
            self.progress_file = Path.home() / '.file_organizer_progress.json'
        
        # Initialize managers (after config overrides)
        self.git_manager = GitManager(self.logger, self.config)
        self.folder_sync = FolderSynchronizer(self.logger, self.git_manager)
        self.background_backup = BackgroundBackup(self.logger, self.config)
        self.content_analyzer = DynamicContentAnalyzer(self.logger, self.config)
        
        # Load progress from previous run (if interrupted)
        self.progress = self._load_progress()
        
        # File classification rules
        self.file_type_mappings = {
            'documents': ['.txt', '.doc', '.docx', '.pdf', '.rtf', '.odt'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
            'videos': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php'],
        }
        
        # Track soft links per file to prevent over-categorization
        self.file_link_counts = defaultdict(int)  # file_path -> count of links created
        self.max_links_per_file = self.config.get('max_soft_links_per_file', 6)
        
        # Track processed files
        self.processed_files = set()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Always validate config if it exists (both test and production modes)
                self._validate_config_sanity(config)
                return config
                
            except json.JSONDecodeError as e:
                # JSON syntax error
                print("\n" + "=" * 70)
                print("INVALID JSON CONFIGURATION")
                print("=" * 70)
                print(f"\nThe file '{self.config_file}' contains invalid JSON:")
                print(f"  {e}")
                print("\nThis is a common mistake when editing configuration files.")
                print("\nTo fix this:")
                print("  1. Check for missing commas, brackets, or quotes")
                print("  2. Use a JSON validator: https://jsonlint.com/")
                print("  3. Or start fresh: cp organizer_config.template.json organizer_config.json")
                print("\n" + "=" * 70 + "\n")
                sys.exit(1)
            except Exception as e:
                # Other config errors (caught by sanity check)
                print("\n" + "=" * 70)
                print("CONFIGURATION VALIDATION FAILED")
                print("=" * 70)
                print(f"\nThe file '{self.config_file}' has configuration errors:")
                print(f"  {e}")
                print("\nPlease fix these errors before running the program.")
                print("\n" + "=" * 70 + "\n")
                sys.exit(1)
        else:
            # Config file doesn't exist
            if self.test_mode:
                # Test mode: OK, will use auto-generated config
                return {}
            else:
                # Production mode: Required!
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
    
    def _validate_config_sanity(self, config: Dict):
        """Validate config for basic sanity (both test and production modes)."""
        errors = []
        warnings = []
        
        # Check for circular references in sync_pairs
        if 'sync_pairs' in config and isinstance(config['sync_pairs'], list):
            drives = config.get('drives', {})
            for i, pair in enumerate(config['sync_pairs']):
                if isinstance(pair, dict) and 'source' in pair and 'target' in pair:
                    source = pair['source']
                    target = pair['target']
                    
                    # Resolve drive placeholders first, then expand and resolve paths
                    try:
                        source_resolved_str = self._resolve_path_with_drives(source, drives)
                        target_resolved_str = self._resolve_path_with_drives(target, drives)
                        source_resolved = str(Path(source_resolved_str).expanduser().resolve())
                        target_resolved = str(Path(target_resolved_str).expanduser().resolve())
                    except Exception as e:
                        errors.append(f"sync_pairs[{i}]: Invalid path - {e}")
                        continue
                    
                    # Check if source and target are the same
                    if source_resolved == target_resolved:
                        errors.append(f"sync_pairs[{i}]: source and target are identical: '{source}' → '{source_resolved}'")
                    
                    # Check if one is a subdirectory of the other (potential circular reference)
                    if source_resolved.startswith(target_resolved + '/') or target_resolved.startswith(source_resolved + '/'):
                        warnings.append(f"sync_pairs[{i}]: potential circular reference - '{source}' and '{target}' are nested")
        
        # Check for output_base conflicts (if output_base exists)
        if 'output_base' in config:
            try:
                output_resolved = str(Path(config['output_base']).expanduser().resolve())
                # Check if output_base is inside any sync pair source or target
                if 'sync_pairs' in config and isinstance(config['sync_pairs'], list):
                    for i, pair in enumerate(config['sync_pairs']):
                        if isinstance(pair, dict) and 'source' in pair and 'target' in pair:
                            try:
                                source_resolved = str(Path(pair['source']).expanduser().resolve())
                                target_resolved = str(Path(pair['target']).expanduser().resolve())
                                if output_resolved.startswith(source_resolved + '/') or output_resolved.startswith(target_resolved + '/'):
                                    warnings.append(f"output_base '{config['output_base']}' is inside sync_pairs[{i}] - this may cause issues")
                            except Exception:
                                pass
            except Exception:
                pass
        
        # Check for reasonable numeric values
        if 'scan_interval' in config:
            interval = config['scan_interval']
            if not isinstance(interval, (int, float)) or interval < 60:
                warnings.append(f"scan_interval ({interval}) is very short - consider at least 300 seconds")
        
        if 'sync_timeout_minutes' in config:
            timeout = config['sync_timeout_minutes']
            if not isinstance(timeout, (int, float)) or timeout < 1:
                warnings.append(f"sync_timeout_minutes ({timeout}) is very short - consider at least 5 minutes")
        
        if 'sync_chunk_concurrency' in config:
            concurrency = config['sync_chunk_concurrency']
            if not isinstance(concurrency, int) or concurrency < 1 or concurrency > 10:
                warnings.append(f"sync_chunk_concurrency ({concurrency}) should be between 1-10")
        
        # Print warnings (non-fatal)
        if warnings:
            print(f"\n⚠️  Configuration warnings in {self.config_file}:")
            for warning in warnings:
                print(f"  • {warning}")
            print()
        
        # Print errors (fatal)
        if errors:
            print(f"\n❌ Configuration errors in {self.config_file}:")
            for error in errors:
                print(f"  • {error}")
            print()
            
            if self.test_mode:
                print("Continuing with auto-generated test config...")
            else:
                print("Please fix these errors before running in production mode.")
                raise ValueError("Configuration validation failed")
    
    def _ensure_config_defaults(self):
        """Ensure all required config keys exist with sensible defaults."""
        # Essential operational settings
        defaults = {
            'flaky_volume_retries': 3,
            'retry_delay': 5,
            'scan_interval': 3600,
            'max_file_size': 104857600,
            'enable_content_analysis': True,
            'enable_duplicate_detection': False,
            'enable_folder_sync': False,
            'enable_git_tracking': False,
            'enable_background_backup': False,
            'use_rsync': True,
            'rsync_checksum_mode': 'timestamp',
            'rsync_size_only': False,
            'rsync_additional_args': [],
            'sync_chunk_subfolders': 30,
            'sync_chunk_concurrency': 1,
            'sync_timeout_minutes': 60,
            'max_drive_usage_percent': 90,
            'max_soft_links_per_file': 6,
            'min_file_size': 1024,
            'exclude_patterns': [
                'node_modules', '.git', '__pycache__', '.DS_Store', '*.pyc',
                '.venv', 'venv', 'env', 'dist', 'build', '.next', '.cache'
            ],
            'exclude_extensions': ['.beam', '.pyc', '.o', '.so'],
            'exclude_folders': [],
            'source_folders': [],
            'sync_pairs': [],
            'backup_directories': [],
            'git_exclude_folders': [],
            'drives': {},
            'output_base': '',
            'backup_drive_path': '',
            'git_user': 'UNCONFIGURED',
            'git_email': 'UNCONFIGURED'
        }
        
        # Apply defaults for missing keys
        for key, default_value in defaults.items():
            if key not in self.config:
                self.config[key] = default_value
        
        # Ensure ml_content_analysis exists with defaults
        if 'ml_content_analysis' not in self.config:
            self.config['ml_content_analysis'] = {
                'enabled': True,
                'min_keyword_frequency': 3,
                'min_category_size': 5,
                'max_categories': 50,
                'min_word_length': 5,
                'stop_words_enabled': True,
                'use_clip': False
            }
        else:
            # Ensure ml_content_analysis has all required sub-keys
            ml_defaults = {
                'enabled': True,
                'min_keyword_frequency': 3,
                'min_category_size': 5,
                'max_categories': 50,
                'min_word_length': 5,
                'stop_words_enabled': True,
                'use_clip': False
            }
            for key, default_value in ml_defaults.items():
                if key not in self.config['ml_content_analysis']:
                    self.config['ml_content_analysis'][key] = default_value
    
    def _resolve_all_paths(self, config: Dict) -> Dict:
        """Resolve all drive placeholders in config paths."""
        resolved = {}
        drives = config.get('drives', {})
        
        # Resolve drives first
        for drive_name, drive_path in drives.items():
            resolved[drive_name] = drive_path
        
        # Resolve other paths
        for key in ['source_folders', 'exclude_folders', 'output_base', 'backup_drive_path']:
            if key in config:
                if isinstance(config[key], list):
                    resolved[key] = [self._resolve_path_with_drives(path, drives) for path in config[key]]
                else:
                    resolved[key] = self._resolve_path_with_drives(config[key], drives)
        
        return resolved
    
    def _resolve_path_with_drives(self, path: str, drives: Dict) -> str:
        """Resolve a path containing drive placeholders."""
        # Check if path starts with a drive name
        for drive_name, drive_path in drives.items():
            # Skip comment keys
            if drive_name.startswith('comment'):
                continue
            # Check if path starts with drive name followed by '/' or is exactly the drive name
            if path.startswith(drive_name + '/') or path == drive_name:
                resolved = path.replace(drive_name, drive_path, 1)
                # Expand ~ if present
                return str(Path(resolved).expanduser())
        
        # No drive placeholder found, just expand ~
        return str(Path(path).expanduser())
    
    def _validate_config_structure(self):
        """Validate configuration file structure before processing (production mode only)."""
        # Skip validation in test mode
        if self.test_mode:
            return
        
        errors = []
        warnings = []
        
        # Check required top-level keys
        required_keys = ['source_folders', 'exclude_folders', 'output_base']
        for key in required_keys:
            if key not in self.config:
                errors.append(f"Missing required key: '{key}'")
        
        # Validate source_folders
        if 'source_folders' in self.config:
            if not isinstance(self.config['source_folders'], list):
                errors.append("'source_folders' must be a list")
            elif len(self.config['source_folders']) == 0:
                warnings.append("'source_folders' is empty - no folders will be scanned")
        
        # Validate sync_pairs structure
        if 'sync_pairs' in self.config:
            if not isinstance(self.config['sync_pairs'], list):
                errors.append("'sync_pairs' must be a list")
            else:
                for i, pair in enumerate(self.config['sync_pairs']):
                    if not isinstance(pair, dict):
                        errors.append(f"sync_pairs[{i}] must be a dictionary")
                        continue
                    
                    # Check if it's just a comment entry (only has 'comment' keys)
                    keys = [k for k in pair.keys() if not k.startswith('comment')]
                    if len(keys) == 0:
                        # Just a comment, skip validation
                        continue
                    
                    # Otherwise, must have source and target
                    if 'source' not in pair:
                        errors.append(f"sync_pairs[{i}] missing 'source' (has: {list(pair.keys())})")
                    if 'target' not in pair:
                        errors.append(f"sync_pairs[{i}] missing 'target' (has: {list(pair.keys())})")
        
        # Validate ml_content_analysis if present
        if 'ml_content_analysis' in self.config:
            ml_config = self.config['ml_content_analysis']
            if not isinstance(ml_config, dict):
                errors.append("'ml_content_analysis' must be a dictionary")
        
        # Print errors and exit if validation failed
        if errors or warnings:
            print("\n" + "=" * 70)
            print("CONFIG VALIDATION ERRORS")
            print("=" * 70)
            print(f"\nConfiguration file: {self.config_file}\n")
            
            if errors:
                print("ERRORS (must fix):")
                for error in errors:
                    print(f"  ✗ {error}")
                print()
            
            if warnings:
                print("WARNINGS:")
                for warning in warnings:
                    print(f"  ⚠ {warning}")
                print()
            
            print("Please fix the configuration file and try again.")
            print("See organizer_config.template.json for a valid example.")
            print("=" * 70 + "\n")
            
            if errors:
                sys.exit(1)
    
    def _check_unconfigured_drives(self):
        """Check for UNCONFIGURED drive placeholders (production mode only)."""
        # Skip validation in test mode - test mode doesn't need drives configured
        if self.test_mode:
            return
        
        # Check if drives are configured
        if 'drives' not in self.config:
            print("\n" + "=" * 70)
            print("ERROR: 'drives' section missing from config file!")
            print("=" * 70)
            print(f"\nConfiguration file: {self.config_file}")
            print("Please add a 'drives' section to your config file.")
            print("\nSee organizer_config.template.json for an example.")
            print("=" * 70 + "\n")
            sys.exit(1)
        
        drives = self.config['drives']
        unconfigured = []
        
        # Check for unconfigured drives
        for drive_name, drive_path in drives.items():
            if isinstance(drive_path, str) and ('UNCONFIGURED' in drive_path or drive_path.strip() == ''):
                unconfigured.append((drive_name, drive_path))
        
        if unconfigured:
            print("\n" + "=" * 70)
            print("ERROR: Drive configuration required!")
            print("=" * 70)
            print(f"\nConfiguration file: {self.config_file}")
            print(f"\nThe following drives are not configured:\n")
            for drive_name, drive_path in unconfigured:
                print(f"  ✗ {drive_name}: {drive_path}")
            print("\nPlease edit the config file and replace the placeholder paths")
            print("with your actual drive paths.")
            print("\nExample:")
            print('  "MAIN_DRIVE": "/Users/yourname"')
            print('  "GOOGLE_DRIVE": "/Users/yourname/Google Drive"')
            print('  "BACKUP_DRIVE": "/Volumes/BackupDrive"')
            print('  "EXTERNAL_DRIVE": "/Volumes/YourDrive"')
            print("\nThen run the program again.")
            print("=" * 70 + "\n")
            sys.exit(1)
    
    def _resolve_drive_placeholders(self):
        """Resolve drive placeholders in config paths (production mode only)."""
        # Resolve drive placeholders in config
        self._resolve_placeholders('source_folders')
        self._resolve_placeholders('exclude_folders')
        self._resolve_placeholder_value('output_base')
        self._resolve_placeholder_value('backup_drive_path')
        
        # Resolve drive placeholders in sync_pairs
        if 'sync_pairs' in self.config:
            drives = self.config.get('drives', {})
            for pair in self.config['sync_pairs']:
                if 'source' in pair:
                    pair['source'] = self._resolve_path_with_drives(pair['source'], drives)
                if 'target' in pair:
                    pair['target'] = self._resolve_path_with_drives(pair['target'], drives)
    
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
        logger = logging.getLogger('file_organizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler - truncate log in production mode
            log_file = Path.home() / '.file_organizer.log'
            log_mode = 'a' if self.test_mode else 'w'  # Append in test mode, truncate in production
            file_handler = logging.FileHandler(log_file, mode=log_mode)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _load_progress(self) -> Dict:
        """Load progress from previous run if it exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    # Check if progress is recent (within last 24 hours)
                    if 'timestamp' in progress:
                        age_hours = (time.time() - progress['timestamp']) / 3600
                        if age_hours < 24:
                            self.logger.info(f"Resuming from previous run (interrupted {age_hours:.1f} hours ago)")
                            return progress
                        else:
                            self.logger.info(f"Previous progress too old ({age_hours:.1f} hours), starting fresh")
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")
        
        return {
            'timestamp': time.time(),
            'current_step': 'scan',
            'scan_folders_completed': [],
            'sync_pairs_completed': [],
            'deduplication_completed': False,
            'backup_completed': False
        }
    
    def _save_progress(self):
        """Save current progress to file."""
        try:
            self.progress['timestamp'] = time.time()
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
            self.logger.debug(f"Progress saved: {self.progress['current_step']}")
        except Exception as e:
            self.logger.warning(f"Could not save progress: {e}")
    
    def _clear_progress(self):
        """Clear progress file after successful completion."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                self.logger.info("Progress cleared - cycle completed successfully")
        except Exception as e:
            self.logger.warning(f"Could not clear progress file: {e}")
        
        # Reset in-memory progress
        self.progress = {
            'timestamp': time.time(),
            'current_step': 'scan',
            'scan_folders_completed': [],
            'sync_pairs_completed': [],
            'deduplication_completed': False,
            'backup_completed': False
        }
    
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
        
        # Check exclude_folders (absolute paths)
        for exclude in self.config.get('exclude_folders', []):
            if path_str.startswith(exclude):
                return True
        
        # Check exclude_patterns (folder/file names anywhere in path)
        for pattern in self.config.get('exclude_patterns', []):
            if pattern in path_str:
                return True
        
        # Check exclude_extensions
        if path.is_file():
            extension = path.suffix.lower()
            if extension in self.config.get('exclude_extensions', []):
                return True
            
            # Check minimum file size (skip tiny files like icons)
            min_size = self.config.get('min_file_size', 0)
            if min_size > 0:
                try:
                    if path.stat().st_size < min_size:
                        return True
                except Exception:
                    pass
        
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
        Valid year range: 1753-2099 (Gregorian calendar in Britain started September 14 1752)
        
        Supports European date format (DD-MM-YY) with preference over American (MM-DD-YY).
        Also supports month names: 11-jul-2020, July-11-2020, July-11-20
        """
        years = []
        MIN_YEAR = 1753  # After Gregorian calendar adoption in Britain
        MAX_YEAR = 2099  # Reasonable future limit
        
        # Month name mapping (abbreviated and full, case-insensitive)
        month_names = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }
        
        # Priority 1: Extract year from filename
        filename = file_path.stem.lower()  # Without extension, lowercase for month matching
        
        # Pattern 1: Month names with day and year (e.g., 11-jul-2020, July-11-2020, July-11-20)
        # Format: DD-MMM-YYYY or DD-MMM-YY (e.g., 11-jul-2020, 11-jul-20)
        match = re.search(r'(\d{1,2})-([a-z]+)-(\d{2,4})', filename)
        if match:
            day, month_str, year = match.groups()
            if month_str in month_names:
                year_int = int(year)
                # Handle 2-digit years
                if len(year) == 2:
                    year_int = 2000 + year_int if year_int < 50 else 1900 + year_int
                if MIN_YEAR <= year_int <= MAX_YEAR:
                    years.append(str(year_int))
                    self.logger.debug(f"Found month-name date {day}-{month_str}-{year} → {year_int}: {file_path.name}")
                    return years
        
        # Format: Month-DD-YYYY or Month-DD-YY (e.g., July-11-2020, July-11-20)
        match = re.search(r'([a-z]+)-(\d{1,2})-(\d{2,4})', filename)
        if match:
            month_str, day, year = match.groups()
            if month_str in month_names:
                year_int = int(year)
                # Handle 2-digit years
                if len(year) == 2:
                    year_int = 2000 + year_int if year_int < 50 else 1900 + year_int
                if MIN_YEAR <= year_int <= MAX_YEAR:
                    years.append(str(year_int))
                    self.logger.debug(f"Found month-name date {month_str}-{day}-{year} → {year_int}: {file_path.name}")
                    return years
        
        # Pattern 2: YYYYMMDD at start (e.g., 20240101-fishing-trip.txt)
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
        
        # Pattern 3: DD-MM-YY or MM-DD-YY at start (e.g., 31-05-18-vienna-shoppe.jpg)
        # Prefer European format unless day > 12 (must be American)
        match = re.match(r'^(\d{2})-(\d{2})-(\d{2})', filename)
        if match:
            first, second, yy = match.groups()
            first_int, second_int = int(first), int(second)
            year_int = 2000 + int(yy) if int(yy) < 50 else 1900 + int(yy)  # 2-digit year conversion
            
            # European format (DD-MM-YY): 31-05-18 → May 31, 2018
            if first_int > 12:  # Must be day (can't be month)
                years.append(str(year_int))
                self.logger.debug(f"Found European date {first}-{second}-{yy} → {year_int}: {file_path.name}")
                return years
            # American format only if second > 12 (must be day)
            elif second_int > 12:  # MM-DD-YY format
                years.append(str(year_int))
                self.logger.debug(f"Found American date {first}-{second}-{yy} → {year_int}: {file_path.name}")
                return years
            # Ambiguous (both <= 12): prefer European (DD-MM-YY)
            else:
                years.append(str(year_int))
                self.logger.debug(f"Found ambiguous date (assuming European) {first}-{second}-{yy} → {year_int}: {file_path.name}")
                return years
        
        # Pattern 4: YYYY-MM-DD anywhere (e.g., backup-2024-01-01.txt)
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
        
        # Pattern 5: YYYY anywhere in filename (e.g., report2024.doc)
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
                    
                    # Check if extracted text is meaningful (not just garbage)
                    if text.strip():
                        # Count words of reasonable length (3+ chars, mostly alphabetic)
                        words = text.split()
                        good_words = [w for w in words if len(w) >= 3 and sum(c.isalpha() for c in w) / len(w) > 0.7]
                        
                        # If fewer than 10 good words per 100 chars, it's probably garbage
                        words_per_100_chars = (len(good_words) * 100) / len(text) if len(text) > 0 else 0
                        
                        if words_per_100_chars < 10:
                            self.logger.info(f"PDF text looks garbled ({words_per_100_chars:.1f} words/100chars), trying OCR")
                            if convert_from_path and pytesseract:
                                ocr_text = self.extract_text_from_scanned_pdf(file_path)
                                # Use OCR if it found more content
                                if len(ocr_text.strip()) > len(text.strip()):
                                    text = ocr_text
                    else:
                        # No text at all, try OCR
                        if convert_from_path and pytesseract:
                            self.logger.info(f"No text in PDF, trying OCR: {file_path.name}")
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
                        # Define categories to check (use keyword-friendly forms)
                        candidate_labels = [
                            "fishing", "fish", "ocean", "sea", "water", "aquarium",
                            "person", "portrait", "woman", "man", "people",
                            "music", "concert", "performance", "singer", "musician",
                            "document", "text", "writing", "book",
                            "nature", "animal", "food", "building", "landscape",
                            "car", "vehicle", "technology", "computer"
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
            # Check if we've already created too many links for this file
            file_key = str(source_path)
            current_count = self.file_link_counts[file_key]
            
            if current_count >= self.max_links_per_file:
                self.logger.debug(f"Skipping link for {source_path.name} - already has {current_count} links (max: {self.max_links_per_file})")
                return False
            
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
            
            # Increment link count for this file
            self.file_link_counts[file_key] += 1
            
            self.logger.debug(f"Created soft link: {target_path} -> {relative_source} (file now has {self.file_link_counts[file_key]} links)")
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
        
        sync_pairs = self.config.get('sync_pairs', [])
        completed_pairs = self.progress.get('sync_pairs_completed', [])
        
        for i, sync_pair in enumerate(sync_pairs):
            # Skip comment-only entries (defensive check - should be caught by validation)
            if 'source' not in sync_pair or 'target' not in sync_pair:
                self.logger.debug(f"Skipping sync_pairs[{i}] - comment-only entry")
                continue
            
            # Skip already completed pairs
            if i in completed_pairs:
                source = sync_pair['source']
                target = sync_pair['target']
                self.logger.info(f"✓ Skipping already completed sync: {source} -> {target}")
                continue
            
            # Paths should already be resolved from drive placeholders, just expand ~ and resolve
            source = Path(sync_pair['source']).expanduser().resolve()
            target = Path(sync_pair['target']).expanduser().resolve()
            
            # Check if source drive/path is available
            source_parent = source
            while source_parent.parent != source_parent and not source_parent.exists():
                source_parent = source_parent.parent
            
            if not source_parent.exists():
                self.logger.warning(f"Source drive/path not available: {source}")
                self.logger.warning(f"Skipping sync pair: {sync_pair.get('source', 'unknown')} -> {sync_pair.get('target', 'unknown')}")
                # Mark as completed so we don't get stuck
                completed_pairs.append(i)
                self.progress['sync_pairs_completed'] = completed_pairs
                self._save_progress()
                continue
            
            if source.exists():
                # Check if we should chunk this sync (large folders)
                chunk_threshold = self.config.get('sync_chunk_subfolders', 30)
                
                if source.is_dir():
                    # Count immediate subfolders
                    try:
                        subfolders = [d for d in source.iterdir() if d.is_dir() and not d.name.startswith('.')]
                        num_subfolders = len(subfolders)
                        
                        if num_subfolders >= chunk_threshold:
                            self.logger.info(f"Large folder detected: {source} has {num_subfolders} subfolders")
                            self.logger.info(f"Syncing in chunks (one subfolder at a time for better progress tracking)")
                            
                            # Sync subfolders with limited concurrency
                            max_workers = max(1, int(self.config.get('sync_chunk_concurrency', 2)))
                            self.logger.info(f"Chunked concurrency: {max_workers}")
                            
                            def sync_one(subfolder_path: Path):
                                try:
                                    if hasattr(self, 'running') and not self.running:
                                        return False
                                    subfolder_target_local = target / subfolder_path.name
                                    return self.folder_sync.sync_directories(subfolder_path, subfolder_target_local, sync_mode='bidirectional')
                                except Exception as e:
                                    self.logger.error(f"Chunk sync failed for {subfolder_path}: {e}")
                                    return False

                            completed = 0
                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                future_to_sub = {executor.submit(sync_one, sf): sf for sf in subfolders}
                                for future in concurrent.futures.as_completed(future_to_sub):
                                    sf = future_to_sub[future]
                                    completed += 1
                                    try:
                                        ok = future.result()
                                        status = "OK" if ok else "FAIL"
                                        self.logger.info(f"  Chunk {completed}/{num_subfolders}: {sf.name} [{status}]")
                                    except Exception as e:
                                        self.logger.error(f"  Chunk {completed}/{num_subfolders}: {sf.name} [ERROR: {e}]")
                            
                            # Sync root level files (if any)
                            root_files = [f for f in source.iterdir() if f.is_file()]
                            if root_files:
                                self.logger.info(f"  Syncing {len(root_files)} root-level files...")
                                # Use bidirectional sync for root files
                                self.folder_sync.sync_directories(source, target, sync_mode='bidirectional')
                        else:
                            # Normal sync for smaller folders
                            self.logger.info(f"Syncing ({i+1}/{len(sync_pairs)}): {source} <-> {target} (bidirectional)")
                            self.folder_sync.sync_directories(source, target, sync_mode='bidirectional')
                    except Exception as e:
                        self.logger.error(f"Error analyzing source folder {source}: {e}")
                        # Fall back to normal sync
                        self.logger.info(f"Syncing ({i+1}/{len(sync_pairs)}): {source} <-> {target} (bidirectional)")
                        self.folder_sync.sync_directories(source, target, sync_mode='bidirectional')
                else:
                    # Not a directory, sync normally
                    self.logger.info(f"Syncing ({i+1}/{len(sync_pairs)}): {source} <-> {target} (bidirectional)")
                    self.folder_sync.sync_directories(source, target, sync_mode='bidirectional')
                
                # Mark this pair as completed
                completed_pairs.append(i)
                self.progress['sync_pairs_completed'] = completed_pairs
                self.progress['current_step'] = 'sync'
                self._save_progress()
            else:
                self.logger.warning(f"Source folder does not exist: {source}")
                # Still mark as completed so we don't get stuck
                completed_pairs.append(i)
                self.progress['sync_pairs_completed'] = completed_pairs
                self._save_progress()
    
    def remove_duplicates_across_system(self) -> None:
        """Find and remove duplicates across all source folders."""
        if not self.config['enable_duplicate_detection']:
            return
        
        # Skip if already completed
        if self.progress.get('deduplication_completed', False):
            self.logger.info("✓ Skipping deduplication - already completed in this run")
            return
        
        self.logger.info("Scanning for duplicate files")
        
        directories = [Path(folder) for folder in self.config['source_folders']]
        duplicates = self.folder_sync.find_duplicates_in_directories(directories)
        
        if duplicates:
            self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
            removed = self.folder_sync.remove_duplicates(duplicates, keep_newest=True)
            self.logger.info(f"Removed {removed} duplicate files")
        
        # Mark as completed
        self.progress['deduplication_completed'] = True
        self.progress['current_step'] = 'deduplication'
        self._save_progress()
    
    def queue_background_backup(self) -> None:
        """Queue important directories for background backup."""
        if not self.config.get('enable_background_backup', False):
            return
        
        # Skip if already completed
        if self.progress.get('backup_completed', False):
            self.logger.info("✓ Skipping backup - already queued in this run")
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
        
        # Mark as completed
        self.progress['backup_completed'] = True
        self.progress['current_step'] = 'backup'
        self._save_progress()
    
    def run_full_cycle(self) -> None:
        """Run a complete organization cycle with dynamic content discovery."""
        self.logger.info("=" * 80)
        self.logger.info("Starting full organization cycle")
        self.logger.info("=" * 80)
        
        # Step 1: Scan and organize files FIRST (gives immediate value before time-consuming operations)
        self.logger.info("Step 1: Scanning and organizing files...")
        
        scan_folders = self.config['source_folders']
        completed_scan_folders = self.progress.get('scan_folders_completed', [])
        
        for i, folder in enumerate(scan_folders):
            if hasattr(self, 'running') and not self.running:
                return
            
            # Skip already completed folders
            if i in completed_scan_folders:
                self.logger.info(f"Skipping already scanned folder: {folder}")
                continue
            
            folder_path = Path(folder)
            if folder_path.exists():
                self.logger.info(f"Scanning folder {i+1}/{len(scan_folders)}: {folder}")
                self._safe_path_operation(self.scan_directory, folder_path)
                
                # Mark this folder as completed
                completed_scan_folders.append(i)
                self.progress['scan_folders_completed'] = completed_scan_folders
                self.progress['current_step'] = 'scan'
                self._save_progress()
        
        # Step 2: Discover content categories based on learned keywords
        if hasattr(self, 'running') and not self.running:
            return
        if self.config.get('ml_content_analysis', {}).get('enabled', True):
            self.logger.info("Step 2: Discovering content categories from analyzed files...")
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
        
        self.logger.info(f"✓ Soft link organization complete! Check {self.config['output_base']}")
        
        # Step 3: Sync configured folders (time-consuming, runs after organization)
        if hasattr(self, 'running') and not self.running:
            return
        self.logger.info("Step 3: Syncing configured folders (this may take a while)...")
        self.sync_configured_folders()
        
        # Step 4: Remove duplicates (time-consuming, runs after organization)
        if hasattr(self, 'running') and not self.running:
            return
        self.logger.info("Step 4: Scanning for duplicate files (this may take a while)...")
        self.remove_duplicates_across_system()
        
        # Step 5: Queue background backup
        if hasattr(self, 'running') and not self.running:
            return
        self.logger.info("Step 5: Queueing background backup...")
        self.queue_background_backup()
        
        # All steps complete - clear progress
        self._clear_progress()
        
        self.logger.info("=" * 80)
        self.logger.info("Full organization cycle complete")
        self.logger.info("=" * 80)
    
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
            "min_keyword_frequency": 2,  # Lower for test mode  
            "min_category_size": 2,  # Lower for test mode - we have small dataset
            "max_categories": 30,  # Increased to see more test categories
            "min_word_length": 4,  # Reduced to 4 for test mode (allows 'peggy' and even 'jazz')
            "stop_words_enabled": True,
            "use_clip": True  # Enable CLIP in test mode to demonstrate fish detection
        }
    }


def check_already_running():
    """Check if file_organizer is already running. Returns (is_running, pids)."""
    try:
        # Get all Python processes running file_organizer.py
        result = subprocess.run(
            ['ps', 'auxw'],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.split('\n')
        pids = []
        current_pid = os.getpid()
        
        for line in lines:
            if 'file_organizer.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    if pid != current_pid:  # Don't count ourselves
                        pids.append(pid)
        
        return len(pids) > 0, pids
    except Exception:
        return False, []


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
    parser.add_argument('--force', action='store_true',
                       help='Kill any existing file_organizer processes before starting')
    parser.add_argument('--no-daemon', action='store_true',
                       help='Do not background the process in production mode (foreground logging)')
    # Internal flag used after relaunching as a background process to suppress console output
    parser.add_argument('--internal-daemon', action='store_true', help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Check if already running (unless creating test environment)
    if not args.create_test:
        is_running, existing_pids = check_already_running()
        if is_running:
            print("\n" + "=" * 70)
            print("WARNING: File Organizer is already running!")
            print("=" * 70)
            print(f"\nFound {len(existing_pids)} existing process(es):")
            for pid in existing_pids:
                try:
                    result = subprocess.run(
                        ['ps', '-p', str(pid), '-o', 'etime=,command='],
                        capture_output=True,
                        text=True
                    )
                    info = result.stdout.strip()
                    print(f"  PID {pid}: {info}")
                except Exception:
                    print(f"  PID {pid}")
            
            print("\nTo stop existing processes:")
            print("  ./manage_organizer.sh stop")
            print("\nOr use --force to kill them automatically:")
            print(f"  python file_organizer.py {' '.join(sys.argv[1:])} --force")
            print("=" * 70 + "\n")
            
            if not args.force:
                sys.exit(1)
            else:
                # Force mode: kill existing processes
                print("--force flag detected: Stopping existing processes...")
                for pid in existing_pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"  Stopped PID {pid}")
                    except Exception as e:
                        print(f"  Could not stop PID {pid}: {e}")
                time.sleep(2)
                print("Continuing with startup...\n")
    
    # Handle test environment creation
    if args.create_test:
        create_test_environment()
        return
    
    # Determine mode
    production_mode = args.REAL
    
    if production_mode and not args.internal_daemon:
        print("=" * 70)
        print("PRODUCTION MODE - Operating on your entire file system")
        print("=" * 70)
        config_file = args.config or 'organizer_config.json'
    else:
        if not args.internal_daemon:
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
        
        # In test mode, still require organizer_config.json for educational setup
        config_file = args.config or 'organizer_config.json'
    
    # In production mode, if not a one-shot scan and not explicitly disabled, daemonize
    if production_mode and not args.scan_once and not args.no_daemon and not args.internal_daemon:
        try:
            background_args = [sys.executable, __file__] + [a for a in sys.argv[1:] if a != '--force'] + ['--internal-daemon']
            subprocess.Popen(background_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
            print("Started File Organizer in background (daemon mode).")
            print("View logs:   ./manage_organizer.sh log")
            print("Stop daemon: ./manage_organizer.sh stop")
            return
        except Exception as e:
            print(f"Failed to start in background: {e}")
            # Fall through and run in foreground

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

