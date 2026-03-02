#!/usr/bin/env python
"""
File Organizer using AI/ML

AI-Generated Code: Created using Cursor AI/ML with human guidance and testing.
Author: Rod McLaughlin (https://github.com/pdxrod)
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
- Background backup to remote drives
- Soft link preservation
- Notification system for offline volumes
- Robust error handling for flaky volumes
- Advanced OCR and AI vision (EasyOCR, CLIP)
"""

import os
import sys

# Suppress macOS MallocStackLogging warnings (harmless but annoying)
# This warning appears when libraries (like PyTorch/NumPy) try to disable malloc stack logging
# that was never enabled. We filter stderr to suppress this specific message.
class MallocStackLoggingFilter:
    """Filter to suppress macOS MallocStackLogging warnings from stderr."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, message):
        # Suppress the specific MallocStackLogging warning
        if 'MallocStackLogging' in message and ("can't turn off" in message or "not enabled" in message):
            return  # Suppress this message
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Apply the filter to stderr before any imports that might trigger the warning
sys.stderr = MallocStackLoggingFilter(sys.stderr)

import time
import hashlib
import logging
import argparse
import signal
import threading
import subprocess
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import mimetypes
import json
import yaml
import re
from collections import defaultdict
import queue
import concurrent.futures

# Suppress PyTorch MPS pin_memory warnings on Apple Silicon (set before torch is ever imported)
warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*MPS.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pinned memory.*', category=UserWarning)

# Third-party imports for AI/ML content analysis
# Note: transformers, torch, easyocr are NOT imported here - they are lazy-loaded when needed
# to avoid 30+ second startup hangs. See get_clip_model() and get_easyocr_reader().
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

# easyocr, transformers, torch: deferred to get_easyocr_reader() / get_clip_model()

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
        """Lazy-load EasyOCR reader (import deferred to avoid startup hang)."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                # Suppress stderr during EasyOCR initialization to hide PyTorch MPS warnings
                import sys
                from io import StringIO
                old_stderr = sys.stderr
                try:
                    sys.stderr = StringIO()
                    self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                finally:
                    sys.stderr = old_stderr
            except ImportError:
                self.logger.debug("EasyOCR not installed")
            except Exception as e:
                self.logger.warning(f"Could not load EasyOCR: {e}")
        return self._easyocr_reader
    
    def get_clip_model(self):
        """Lazy-load CLIP model (import deferred to avoid startup hang)."""
        if self._clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.logger.info("Loading CLIP model (first time only)...")
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.logger.info("CLIP loaded successfully")
            except ImportError:
                self.logger.debug("CLIP/transformers not installed")
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
        
        # Split on common delimiters: spaces, dashes, underscores, dots, commas, @ symbols
        # This handles filenames like "31-05-18-vienna-shoppe-kirche.jpg" and social media handles like "@jack"
        text = re.sub(r'[-_.,/\\()[\]{}@]', ' ', text)
        
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


class SemanticCategorizer:
    """Groups keywords into broad semantic categories using sentence embeddings.
    
    Uses a small transformer model (all-MiniLM-L6-v2, ~80MB) to understand that
    'london' belongs to 'Cities & Places', 'schnitzel' to 'Food & Cuisine', etc.
    Model is lazy-loaded on first use and cached locally by transformers.
    """
    
    BROAD_CATEGORIES = {
        "People & Personalities": "people persons individuals names celebrities politicians leaders figures profiles biography",
        "Cities & Places": "cities towns places locations neighborhoods landmarks urban metropolitan capital borough",
        "Countries & Nations": "countries nations states regions continents borders territories international sovereign",
        "Politics & Government": "politics government political policy election parliament congress senate legislation diplomacy",
        "Legal & Justice": "legal law court judge attorney lawyer lawsuit verdict trial testimony prosecution",
        "Crime & Investigation": "crime criminal investigation detective suspect fraud corruption evidence forensic police",
        "Finance & Economics": "finance money budget economy tax salary income payment banking investment market",
        "Technology & Software": "technology software programming computer code algorithm digital engineering developer",
        "Science & Research": "science research study experiment data analysis theory laboratory discovery physics biology",
        "Health & Medicine": "health medical hospital doctor medicine disease treatment wellness pharmacy therapy",
        "Education & Academia": "education school university college learning teaching professor student academic degree",
        "Travel & Tourism": "travel trip journey flight hotel vacation tourism destination adventure abroad passport",
        "Food & Cuisine": "food cooking recipe restaurant cuisine dining meal ingredients chef culinary kitchen",
        "Music & Performance": "music song band concert instrument guitar piano drum singing performer melody album",
        "Art & Culture": "art painting sculpture gallery museum creative artistic culture exhibition craft",
        "Sports & Fitness": "sports athletics football soccer basketball tennis game match competition fitness exercise",
        "Nature & Wildlife": "nature environment wildlife animal forest ocean river mountain ecology conservation",
        "Business & Commerce": "business company corporate enterprise startup office industry commerce trade professional",
        "Media & Journalism": "media news journalism press reporter newspaper editorial broadcast interview article",
        "Military & Defense": "military army navy defense forces combat war weapons security veteran intelligence",
        "Religion & Faith": "religion church faith spiritual temple mosque worship prayer belief theology sacred",
        "Entertainment & Film": "entertainment movie film television series show cinema actor theater drama comedy",
        "Family & Relationships": "family home children parents marriage relationship household parenting domestic",
        "Communication & Writing": "email letter message writing communication correspondence document memo note",
        "Transportation": "vehicle car transport train airplane bus shipping driving road highway aviation",
        "Real Estate & Architecture": "property house building apartment architecture construction estate housing land",
        "History & Heritage": "history historical ancient heritage archive century era past legacy memorial revolution",
        "Photography & Visual": "photo photograph camera image portrait landscape snapshot lens exposure studio",
        "Social Media & Internet": "social media youtube twitter instagram facebook online platform website blog post",
        "Climate & Weather": "weather climate temperature rain snow storm hurricane season atmospheric warming",
    }
    
    def __init__(self, logger: logging.Logger, config: Dict):
        self.logger = logger
        self.config = config
        self._model = None
        self._tokenizer = None
        self._category_embeddings = None
        self._category_names = list(self.BROAD_CATEGORIES.keys())
    
    def _get_model(self):
        """Lazy-load sentence embedding model using the transformers library."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                self.logger.info("Loading semantic categorization model (first time downloads ~80MB)...")
                import sys
                from io import StringIO
                old_stderr = sys.stderr
                try:
                    sys.stderr = StringIO()
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._model = AutoModel.from_pretrained(model_name)
                finally:
                    sys.stderr = old_stderr
                self.logger.info("Semantic categorization model loaded")
            except ImportError:
                self.logger.warning("transformers/torch not installed — semantic categories disabled")
            except Exception as e:
                self.logger.warning(f"Could not load semantic model: {e}")
        return self._model
    
    def _embed(self, texts: list):
        """Compute L2-normalized embeddings for a batch of texts."""
        import torch
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self._model(**encoded)
        mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)
    
    def _get_category_embeddings(self):
        if self._category_embeddings is None:
            self._category_embeddings = self._embed(list(self.BROAD_CATEGORIES.values()))
        return self._category_embeddings
    
    def classify_keywords(self, keywords: set) -> Dict[str, tuple]:
        """Classify keywords into semantic categories via cosine similarity.
        
        Returns dict: keyword -> (category_name, confidence_score)
        """
        import torch
        if not self._get_model() or not keywords:
            return {}
        
        keyword_list = list(keywords)
        keyword_embs = self._embed(keyword_list)
        cat_embs = self._get_category_embeddings()
        
        similarities = torch.mm(keyword_embs, cat_embs.t())
        threshold = self.config.get('semantic_confidence_threshold', 0.35)
        
        result = {}
        for i, kw in enumerate(keyword_list):
            best_idx = similarities[i].argmax().item()
            best_score = similarities[i][best_idx].item()
            if best_score >= threshold:
                result[kw] = (self._category_names[best_idx], best_score)
        
        return result
    
    def group_files(self, file_keywords_map: Dict[str, set]) -> Dict[str, set]:
        """Group files into semantic categories based on their keywords.
        
        Args:
            file_keywords_map: {file_path_str: set_of_keywords}
        
        Returns:
            {semantic_category_name: set_of_file_path_strings}
        """
        all_keywords = set()
        for kws in file_keywords_map.values():
            all_keywords.update(kws)
        
        if not all_keywords:
            return {}
        
        classifications = self.classify_keywords(all_keywords)
        if not classifications:
            return {}
        
        groups = defaultdict(set)
        for file_path, kws in file_keywords_map.items():
            for kw in kws:
                if kw in classifications:
                    cat_name, _ = classifications[kw]
                    groups[cat_name].add(file_path)
        
        for cat, files in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
            self.logger.info(f"  Semantic category '{cat}': {len(files)} files")
        
        return dict(groups)


class FolderSynchronizer:
    """Synchronizes folders and removes duplicates."""
    
    def __init__(self, logger: logging.Logger, config: Dict):
        self.logger = logger
        self.config = config
    
    def _get_organised_base(self) -> Path:
        """Get the organised_base path from config, with default fallback."""
        backup_base = self.config.get('softlink_backup_base', '~/organised')
        # Expand ~ to home directory
        if backup_base.startswith('~'):
            backup_base = str(Path.home()) + backup_base[1:]
        return Path(backup_base)
    
    def _shorten_path(self, path: Path, base_paths: List[Path] = None, include_filename: bool = True) -> str:
        """
        Shorten a path for compact display.
        Shows: Users/.../subdir/filename or Volumes/Drive/.../subdir/filename
        Just enough start to identify the drive, then relevant subdirectory (and optionally filename).
        Removes leading slash for cleaner output.
        """
        path_str = str(path)
        
        # If path is short enough, return as-is (without leading slash)
        if len(path_str) <= 50:
            return path_str.lstrip('/')
        
        parts = path.parts
        if len(parts) == 0:
            return path_str.lstrip('/')
        
        # Determine how much of the start to show (skip root slash)
        # For /Users paths: show Users/ (1 part after root)
        # For /Volumes paths: show Volumes/DriveName/ (2 parts after root to identify drive)
        start_parts = []
        
        if parts[0] == '/Volumes' and len(parts) >= 3:
            # For external drives, show Volumes/DriveName to identify which drive
            start_parts.extend(parts[1:3])  # Skip root, include Volumes and drive name
        elif parts[0] == '/' and len(parts) >= 2:
            # For /Users or other root paths, show first part after root
            start_parts.append(parts[1])
        
        # Get the last 2-3 parts (subdirectory and optionally filename)
        # This shows the relevant context around the file
        end_parts = []
        if include_filename and len(parts) >= 1:
            end_parts.append(parts[-1])  # filename
        if len(parts) >= 2:
            end_parts.insert(0, parts[-2])  # parent dir
        if len(parts) >= 3 and len(end_parts) < 3:
            # If we have room, include grandparent for more context
            end_parts.insert(0, parts[-3])
        
        # Build shortened path
        start_str = '/'.join(start_parts)
        end_str = '/'.join(end_parts)
        
        # Check if start and end overlap
        total_start_end_parts = len(start_parts) + len(end_parts)
        if total_start_end_parts >= len(parts):
            # Overlap - just show start + ... + (filename or last part)
            if include_filename and len(parts) >= 1:
                return f"{start_str}/.../{parts[-1]}"
            elif len(parts) >= 2:
                return f"{start_str}/.../{parts[-2]}/"
            else:
                return start_str
        
        return f"{start_str}/.../{end_str}"
    
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
    
    def remove_duplicates(self, duplicates: Dict[str, List[Path]], keep_newest: bool = True, dry_run: bool = True) -> int:
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
                if dry_run:
                    self.logger.info(f"[DRY RUN] Would remove duplicate: {duplicate_file}")
                    removed_count += 1
                else:
                    try:
                        duplicate_file.unlink()
                        self.logger.info(f"Removed duplicate: {duplicate_file}")
                        removed_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to remove {duplicate_file}: {e}")

        return removed_count
    
    def _matches_pattern(self, path: Path, patterns: List[str]) -> bool:
        """
        Check if path matches any pattern in the list (supports wildcards).
        For folder patterns, only matches against the folder NAME, not the full path.
        This prevents "env" from matching "environments".
        """
        import fnmatch
        
        # Get just the folder/file name (last component of the path)
        folder_name = path.name
        
        for pattern in patterns:
            # Check if pattern contains wildcards
            if '*' in pattern or '?' in pattern:
                # Use fnmatch for wildcard patterns
                # Only check the folder name (exact match on the component name)
                if fnmatch.fnmatch(folder_name, pattern):
                    return True
            else:
                # Exact match on folder name (not substring)
                # This prevents "env" from matching "environments"
                if folder_name == pattern:
                    return True
        
        return False
    
    def _should_process_excluded_folders(self, directory: Path) -> bool:
        """
        Determine if excluded folders should be processed in this directory.
        Returns False for cloud drives and external volumes.
        """
        try:
            dir_str = str(directory.resolve())
            
            # Don't process cloud drives
            if '/ProtonDrive' in dir_str or '/GoogleDrive' in dir_str:
                return False
            
            # Don't process external volumes (mounted at /Volumes/)
            if dir_str.startswith('/Volumes/'):
                return False
            
            # Process local drives (under /Users, etc.)
            return True
        except Exception:
            # If we can't determine, err on the side of caution and don't process
            return False
    
    def _process_excluded_folders_in_directory(self, directory: Path, organised_base: Path, max_depth: int = 20) -> None:
        """
        Recursively process excluded folders in a directory:
        - softlink_folder_patterns: Backup to softlink_backup_base and replace with soft links
        - empty_folder_patterns: Delete contents, leave empty folders
        
        Processes directories recursively to find all excluded folders.
        """
        if max_depth <= 0:
            return
            
        config = self.config
        softlink_patterns = config.get('softlink_folder_patterns', [])
        empty_patterns = config.get('empty_folder_patterns', [])
        
        if not directory.exists() or not directory.is_dir():
            return
        
        # Skip if this directory itself is in softlink_backup_base
        try:
            if directory.is_relative_to(organised_base):
                return
        except ValueError:
            pass
        
        try:
            for item in directory.iterdir():
                if not item.is_dir():
                    continue
                
                # Skip if already a symlink (already processed)
                if item.is_symlink():
                    continue
                
                # Check if this folder matches softlink_folder_patterns
                if softlink_patterns and self._matches_pattern(item, softlink_patterns):
                    self._backup_and_link_folder(item, organised_base)
                # Check if this folder matches empty_folder_patterns
                elif empty_patterns and self._matches_pattern(item, empty_patterns):
                    self._empty_folder(item)
                else:
                    # Recursively process subdirectories
                    self._process_excluded_folders_in_directory(item, organised_base, max_depth - 1)
        except PermissionError:
            self.logger.warning(f"Permission denied processing excluded folders in {directory}")
        except Exception as e:
            self.logger.error(f"Error processing excluded folders in {directory}: {e}")
    
    def _backup_and_link_folder(self, folder: Path, organised_base: Path) -> None:
        """
        Backup a folder to softlink_backup_base maintaining full source path structure,
        then replace the original with a soft link.
        
        Example: /Users/yourname/dev/bash/.git 
        -> Backup to: {softlink_backup_base}/Users/yourname/dev/bash/.git
        -> Replace with: /Users/yourname/dev/bash/.git (soft link)
        """
        try:
            # Skip if already a symlink
            if folder.is_symlink():
                return
            
            # Calculate backup path maintaining full source structure
            # e.g., /Users/yourname/dev/bash/.git -> {softlink_backup_base}/Users/yourname/dev/bash/.git
            # Use absolute path to preserve full structure
            folder_abs = folder.resolve()
            # Remove leading slash and join with organised_base
            folder_str = str(folder_abs)
            if folder_str.startswith('/'):
                folder_str = folder_str[1:]  # Remove leading slash
            backup_path = organised_base / folder_str
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If backup already exists, check if we need to update the link
            if backup_path.exists():
                self.logger.debug(f"Backup already exists for {folder} at {backup_path}")
                # Still replace with link if not already a link
                if not folder.is_symlink():
                    # Remove original folder (but keep contents safe in backup)
                    if folder.exists() and folder.is_dir():
                        try:
                            shutil.rmtree(folder)  # Use rmtree for directories
                        except Exception as e:
                            self.logger.warning(f"Could not remove folder {folder}: {e}")
                            return
                    folder.symlink_to(backup_path)
                    self.logger.info(f"Replaced {folder} with soft link to {backup_path}")
                elif folder.is_symlink():
                    # Check if link points to correct location
                    try:
                        current_target = folder.readlink()
                        if current_target != backup_path:
                            folder.unlink()
                            folder.symlink_to(backup_path)
                            self.logger.info(f"Updated {folder} soft link to point to {backup_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not read/update symlink {folder}: {e}")
                return
            
            # Copy folder contents to backup location
            if folder.exists():
                try:
                    # Check if folder has contents
                    has_contents = False
                    try:
                        next(folder.iterdir())
                        has_contents = True
                    except StopIteration:
                        pass
                    
                    if has_contents:
                        shutil.copytree(folder, backup_path, dirs_exist_ok=True)
                        self.logger.info(f"Backed up {folder} to {backup_path}")
                    else:
                        # Create empty backup directory
                        backup_path.mkdir(parents=True, exist_ok=True)
                        self.logger.debug(f"Created empty backup directory {backup_path} for {folder}")
                except Exception as e:
                    self.logger.error(f"Failed to copy folder {folder} to backup: {e}")
                    return
            
            # Remove original and create soft link
            if folder.exists() and not folder.is_symlink() and folder.is_dir():
                try:
                    shutil.rmtree(folder)  # Use rmtree for directories, not unlink
                except Exception as e:
                    self.logger.error(f"Could not remove folder {folder} before creating symlink: {e}")
                    return
            
            # Create the symlink
            try:
                folder.symlink_to(backup_path)
                self.logger.info(f"Replaced {folder} with soft link to {backup_path}")
            except FileExistsError:
                # Folder already exists (might be a broken symlink or something)
                try:
                    if folder.is_symlink():
                        folder.unlink()
                    folder.symlink_to(backup_path)
                    self.logger.info(f"Replaced existing {folder} with soft link to {backup_path}")
                except Exception as e:
                    self.logger.error(f"Could not create symlink {folder} -> {backup_path}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to backup and link folder {folder}: {e}")
    
    def _empty_folder(self, folder: Path) -> None:
        """Delete all contents of a folder, leaving it empty."""
        try:
            if not folder.exists() or not folder.is_dir():
                return
            
            # Skip if it's a symlink
            if folder.is_symlink():
                return
            
            # Delete all contents
            for item in folder.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    self.logger.warning(f"Could not delete {item} in {folder}: {e}")
            
            self.logger.debug(f"Emptied folder {folder}")
        except Exception as e:
            self.logger.error(f"Failed to empty folder {folder}: {e}")
    
    # Cloud storage mount directories and their conflict copies must never be synced.
    _CLOUD_STORAGE_PATTERNS = [
        'GoogleDrive-', 'ProtonDrive-', 'OneDrive-', 'Dropbox-',
        'iCloud Drive', 'CloudStorage',
        '.Trash', '.Trashes',
    ]

    def should_exclude_from_sync(self, path: Path, sync_excludes: List[str]) -> bool:
        """Check if path should be excluded from sync.
        
        Supports wildcard patterns (e.g., '.tmp*', '*.pyc') and simple substring matching.
        Checks both the full path and individual path components (directory/file names).
        Always excludes softlink_backup_base to prevent copying the backup location.
        """
        import fnmatch
        
        path_str = str(path)
        path_parts = path.parts
        
        # CRITICAL: Always exclude softlink_backup_base (where excluded folders are backed up)
        organised_path = self._get_organised_base()
        try:
            if path.is_relative_to(organised_path) or organised_path in path.parents:
                return True
        except ValueError:
            if 'organised' in path_str and str(organised_path) in path_str:
                return True
        
        # Never sync cloud storage mount dirs or their conflict copies
        for cloud_pat in self._CLOUD_STORAGE_PATTERNS:
            for part in path_parts:
                if cloud_pat in part:
                    return True
        
        for exclude in sync_excludes:
            # Check if pattern contains wildcards
            if '*' in exclude or '?' in exclude:
                # Use fnmatch for wildcard patterns
                # Check individual path components (directory/file names) - most common case
                # e.g., '.tmp*' should match '.tmp.driveupload' directory name
                for part in path_parts:
                    if fnmatch.fnmatch(part, exclude):
                        return True
                # Also check if pattern matches anywhere in the full path
                # This handles cases like '*node_modules*'
                if fnmatch.fnmatch(path_str, f"*{exclude}*"):
                    return True
            else:
                # Simple substring match for patterns without wildcards
                # Check if pattern appears anywhere in the path
                if exclude in path_str:
                    return True
        
        return False
    
    def _is_cloud_fuse_path(self, path: Path) -> bool:
        """Return True if path is on a cloud FUSE mount (ProtonDrive, GoogleDrive, etc.)."""
        path_str = str(path)
        return '/ProtonDrive' in path_str or '/GoogleDrive' in path_str

    def _get_fuse_rsync_args(self) -> List[str]:
        """
        Return rsync args that prevent failures on FUSE/cloud mounts.

        FUSE filesystems (ProtonDrive, GoogleDrive) cannot accept Unix permission or
        ownership changes; rsync exits 23/24 on every run without these flags.
        --modify-window=2 tolerates the coarser timestamp precision common on FUSE/FAT.
        """
        return [
            '--omit-dir-times',  # Don't sync directory mtimes (FUSE usually ignores them)
            '--no-perms',        # Don't sync file permissions
            '--no-owner',        # Don't sync file ownership
            '--no-group',        # Don't sync group
            '--modify-window=2', # 2-second mtime tolerance (FUSE/FAT timestamp precision)
        ]

    def sync_with_rsync(self, source: Path, target: Path, sync_mode: str, sync_excludes: List[str]) -> bool:
        """Fast synchronization using rsync."""
        try:
            rsync_mode = self.config.get('rsync_checksum_mode', 'checksum')
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
            if self.config.get('rsync_size_only', False):
                cmd.append('--size-only')

            # Auto-add FUSE-compatible flags for cloud mounts (ProtonDrive, GoogleDrive).
            # Without these, rsync exits 23/24 every run because FUSE rejects chmod/chown.
            if self._is_cloud_fuse_path(source) or self._is_cloud_fuse_path(target):
                self.logger.info("Cloud FUSE mount detected — adding FUSE-compatible rsync flags")
                for arg in self._get_fuse_rsync_args():
                    cmd.append(arg)

            # Add additional args to reduce metadata ops on cloud/FUSE targets if configured
            # Example: --omit-dir-times --no-perms --no-group --no-owner --delete-after
            additional_args = self.config.get('rsync_additional_args', []) or []
            for extra in additional_args:
                if isinstance(extra, str) and extra.strip():
                    cmd.append(extra.strip())

            # Disable mmap on macOS or when explicitly configured to avoid
            # "mmap: Operation canceled" errors during reads.
            if self.config.get('rsync_disable_mmap', False):
                cmd.append('--no-mmap')
            
            # Add exclusions
            for exclude in sync_excludes:
                cmd.extend(['--exclude', exclude])
            
            # Add source and target (trailing slash important for rsync!)
            cmd.append(f'{source}/')
            cmd.append(f'{target}/')
            
            # Run rsync with timeout to prevent infinite hangs
            timeout_minutes = self.config.get('sync_timeout_minutes', 30)
            timeout_seconds = timeout_minutes * 60
            
            self.logger.info(f"Timeout: {timeout_minutes} min | Excluding: {', '.join(sync_excludes[:5])}{'...' if len(sync_excludes) > 5 else ''}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
                if result.returncode != 0 and '--no-mmap' in cmd:
                    stderr_lower = (result.stderr or '').lower()
                    if 'no-mmap' in stderr_lower and 'unknown option' in stderr_lower:
                        self.logger.warning("rsync does not support --no-mmap; retrying without it")
                        cmd = [arg for arg in cmd if arg != '--no-mmap']
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
                return True
            else:
                self.logger.error(f"rsync failed with exit code {result.returncode}")
                self.logger.error(f"rsync error: {result.stderr[:500]}")
                return False
                
        except Exception as e:
            self.logger.error(f"rsync sync failed: {e}")
            return False

    def sync_with_rsync_bidirectional(self, folder_a: Path, folder_b: Path, sync_excludes: List[str]) -> bool:
        """
        Bidirectional sync using two rsync passes (vastly faster than Python rglob on FUSE mounts).

        Strategy — run rsync twice, both with --update so only the newer file wins:
          Pass 1: A → B  (A-only files and newer-A files flow to B)
          Pass 2: B → A  (B-only files and newer-B files flow to A)

        No --delete flag in either pass: files that legitimately exist only in one
        folder are preserved and synced to the other side in the opposite pass.

        This matches _sync_bidirectional semantics but delegates the filesystem walk
        to rsync's C implementation, which is critical for slow FUSE mounts where
        every stat() call is a network round-trip.
        """
        is_cloud = self._is_cloud_fuse_path(folder_a) or self._is_cloud_fuse_path(folder_b)
        timeout_minutes = self.config.get('sync_timeout_minutes', 60)
        timeout_seconds = timeout_minutes * 60

        def _build_cmd(src: Path, dst: Path) -> List[str]:
            cmd = ['rsync', '-avh', '--update', '--stats']
            if self.config.get('rsync_size_only', False):
                cmd.append('--size-only')
            if is_cloud:
                for arg in self._get_fuse_rsync_args():
                    cmd.append(arg)
            for extra in (self.config.get('rsync_additional_args', []) or []):
                if isinstance(extra, str) and extra.strip():
                    cmd.append(extra.strip())
            if self.config.get('rsync_disable_mmap', False):
                cmd.append('--no-mmap')
            for excl in sync_excludes:
                cmd.extend(['--exclude', excl])
            cmd.append(f'{src}/')
            cmd.append(f'{dst}/')
            return cmd

        def _run_pass(src: Path, dst: Path, label: str) -> bool:
            cmd = _build_cmd(src, dst)
            mode_tag = 'cloud-FUSE' if is_cloud else 'local'
            self.logger.info(f"rsync bidirectional {label} ({mode_tag}): {src} → {dst}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
                # Retry without --no-mmap if this rsync version doesn't support it
                if result.returncode != 0 and '--no-mmap' in cmd:
                    stderr_lower = (result.stderr or '').lower()
                    if 'no-mmap' in stderr_lower and 'unknown option' in stderr_lower:
                        self.logger.warning("rsync does not support --no-mmap; retrying without it")
                        cmd = [a for a in cmd if a != '--no-mmap']
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self.logger.error(f"rsync bidirectional {label} timed out after {timeout_minutes} min")
                return False

            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'transferred:' in line or 'speedup' in line:
                        self.logger.info(f"rsync {label}: {line.strip()}")
                return True
            elif result.returncode in [23, 24]:
                if is_cloud:
                    self.logger.info(
                        f"rsync {label} partial success (exit {result.returncode}) — "
                        f"expected on FUSE mounts; files transferred successfully"
                    )
                    if 'fchmodat' in result.stderr or 'permission' in result.stderr.lower():
                        self.logger.debug("Permission errors are harmless on cloud FUSE mounts")
                else:
                    self.logger.warning(f"rsync {label} partial success (exit {result.returncode})")
                    for line in result.stderr.split('\n')[:5]:
                        if line.strip():
                            self.logger.warning(f"  {line.strip()}")
                return True
            else:
                self.logger.error(
                    f"rsync {label} failed (exit {result.returncode}): {result.stderr[:300]}"
                )
                return False

        ok_ab = _run_pass(folder_a, folder_b, 'A→B')
        ok_ba = _run_pass(folder_b, folder_a, 'B→A')
        return ok_ab and ok_ba

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
            
            # If parent doesn't exist, the drive is offline
            if not target_parent.exists():
                self.logger.warning(f"Target drive offline or not accessible: {target}")
                self.logger.warning(f"Skipping sync - drive at {target_parent} not found")
                return False
            
            # CRITICAL: Check if target path contains "MAIN_DRIVE" as a literal directory name
            # This prevents creating literal MAIN_DRIVE directories
            target_str = str(target)
            if 'MAIN_DRIVE' in target_str:
                # Check if MAIN_DRIVE appears as a directory component (not just in a filename)
                target_parts = target.parts
                if 'MAIN_DRIVE' in target_parts:
                    # This is a bug - MAIN_DRIVE should have been resolved
                    error_msg = (f"CRITICAL BUG DETECTED: Target path contains literal 'MAIN_DRIVE' directory: {target}. "
                               f"This will create a literal MAIN_DRIVE folder. Path parts: {target_parts}. "
                               f"Please check your drives configuration.")
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Test write access by actually trying to create a test file
            # This is more reliable than os.access() on macOS external drives
            try:
                # Ensure target directory exists for the test
                target.mkdir(parents=True, exist_ok=True)
                # Try to create a temporary test file
                test_file = target / f".write_test_{os.getpid()}"
                test_file.touch()
                test_file.unlink()  # Clean up immediately
            except (OSError, PermissionError) as e:
                self.logger.warning(f"Target drive not writable: {target}")
                self.logger.warning(f"Skipping sync - no write access to {target}: {e}")
                return False
            
            # Check disk space on target drive
            max_usage = self.config.get('max_drive_usage_percent', 90)
            try:
                stat = os.statvfs(target_parent)
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_bfree * stat.f_frsize
                used_percent = ((total - free) / total) * 100
                
                if used_percent >= max_usage:
                    self.logger.warning(f"Target drive {target_parent} is {used_percent:.1f}% full (max: {max_usage}%)")
                    self.logger.warning(f"Skipping sync to prevent filling drive")
                    return False
            except Exception as e:
                self.logger.debug(f"Could not check disk space: {e}")
                
        except Exception as e:
            self.logger.warning(f"Cannot access target drive for {target}: {e}")
            return False
        
        # CRITICAL: Check if target path contains "MAIN_DRIVE" as a literal directory name
        target_str = str(target)
        if 'MAIN_DRIVE' in target_str:
            target_parts = target.parts
            if 'MAIN_DRIVE' in target_parts:
                # This is a bug - MAIN_DRIVE should have been resolved
                error_msg = (f"CRITICAL BUG DETECTED: Target path contains literal 'MAIN_DRIVE' directory: {target}. "
                           f"This will create a literal MAIN_DRIVE folder. Path parts: {target_parts}. "
                           f"Please check your drives configuration.")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Create target directory
        try:
            target.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.logger.error(f"Cannot create target directory {target}: {e}")
            self.logger.warning(f"Skipping sync - target may be on offline drive")
            return False
        
        # Get sync exclusions from config - combine all exclude patterns
        exclude_patterns = self.config.get('exclude_patterns', [])
        softlink_patterns = self.config.get('softlink_folder_patterns', [])
        empty_patterns = self.config.get('empty_folder_patterns', [])
        all_exclude_patterns = exclude_patterns + softlink_patterns + empty_patterns
        
        # Process excluded folders before syncing (only on local drives, not cloud/external)
        # This backs up softlink_folder_patterns to softlink_backup_base and empties empty_folder_patterns
        organised_base = self._get_organised_base()
        
        # Only process excluded folders on local/main drive, not on cloud or external drives
        # Cloud/external drives shouldn't have their folders backed up to softlink_backup_base
        if self._should_process_excluded_folders(source):
            self._process_excluded_folders_in_directory(source, organised_base)
        if self._should_process_excluded_folders(target):
            self._process_excluded_folders_in_directory(target, organised_base)
        
        use_rsync = self.config.get('use_rsync', True)

        # For bidirectional sync, prefer two-pass rsync (--update each direction).
        # This is vastly faster than Python rglob on FUSE mounts like ProtonDrive because
        # the filesystem walk is done in C and stats are batched rather than one per call.
        if sync_mode == 'bidirectional':
            if use_rsync and shutil.which('rsync'):
                if self.sync_with_rsync_bidirectional(source, target, all_exclude_patterns):
                    return True
                self.logger.warning("rsync bidirectional failed — falling back to Python sync")
            return self._sync_bidirectional(source, target, all_exclude_patterns, [source, target])

        # For one-way sync, try rsync first (much faster)
        if use_rsync and shutil.which('rsync'):
            return self.sync_with_rsync(source, target, sync_mode, all_exclude_patterns)
        
        # Fallback to manual Python sync (if rsync not available)
        return self._sync_one_way(source, target, sync_mode, all_exclude_patterns)
    
    def _sync_bidirectional(self, folder_a: Path, folder_b: Path, exclude_patterns: List[str], base_paths: List[Path] = None) -> bool:
        """
        Bidirectional sync between two folders (A and B are interchangeable):
        1. If file exists in A but not in B → copy from A to B
        2. If file exists in B but not in A → copy from B to A
        3. If file exists in both but A is newer → copy from A to B
        4. If file exists in both but B is newer → copy from B to A
        """
        if base_paths is None:
            base_paths = [folder_a, folder_b]
        
        copied_a_to_b = 0
        copied_b_to_a = 0
        
        try:
            # Collect all files from both directories
            files_a = {}
            files_b = {}
            
            # Scan folder A
            if folder_a.exists():
                for item in folder_a.rglob('*'):
                    if item.is_file() and not self.should_exclude_from_sync(item, exclude_patterns):
                        relative_path = item.relative_to(folder_a)
                        files_a[str(relative_path)] = item
            
            # Scan folder B
            if folder_b.exists():
                for item in folder_b.rglob('*'):
                    if item.is_file() and not self.should_exclude_from_sync(item, exclude_patterns):
                        relative_path = item.relative_to(folder_b)
                        files_b[str(relative_path)] = item
            
            # Process all files (union of A and B)
            all_files = set(files_a.keys()) | set(files_b.keys())
            
            for relative_path_str in all_files:
                relative_path = Path(relative_path_str)
                file_a = files_a.get(relative_path_str)
                file_b = files_b.get(relative_path_str)
                
                # Determine copy direction
                copy_from_b = False
                
                if file_a is None and file_b is not None:
                    # File exists only in B → copy from B to A
                    copy_from_b = True
                elif file_a is not None and file_b is None:
                    # File exists only in A → copy from A to B
                    copy_from_b = False
                elif file_a is not None and file_b is not None:
                    # File exists in both → compare timestamps
                    try:
                        mtime_a = file_a.stat().st_mtime
                        mtime_b = file_b.stat().st_mtime
                        # If B is newer, copy from B to A
                        copy_from_b = (mtime_b > mtime_a)
                    except Exception as e:
                        self.logger.warning(f"Could not compare timestamps for {relative_path}: {e}")
                        # Default: copy from A to B
                        copy_from_b = False
                
                # Perform the copy
                if copy_from_b:
                    # Copy from B to A
                    dest_file = folder_a / relative_path
                    # CRITICAL: Check for MAIN_DRIVE in path before creating directory
                    if 'MAIN_DRIVE' in dest_file.parts:
                        error_msg = (f"CRITICAL BUG: Attempting to create directory with MAIN_DRIVE: {dest_file}. "
                                   f"Path parts: {dest_file.parts}")
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if file_b.is_symlink():
                            link_target = os.readlink(file_b)
                            if dest_file.exists() or dest_file.is_symlink():
                                dest_file.unlink()
                            dest_file.symlink_to(link_target)
                            self.logger.debug(f"Copied symlink: {file_b} -> {dest_file}")
                        else:
                            shutil.copy2(file_b, dest_file)
                            # Use shortened paths for compact output
                            # Source includes filename, destination shows directory only
                            short_src = self._shorten_path(file_b, base_paths, include_filename=True)
                            short_dest = self._shorten_path(dest_file, base_paths, include_filename=False)
                            self.logger.info(f"{short_src} → {short_dest}")
                        copied_b_to_a += 1
                    except Exception as e:
                        self.logger.error(f"Failed to copy {file_b} to {dest_file}: {e}")
                else:
                    # Copy from A to B
                    dest_file = folder_b / relative_path
                    # CRITICAL: Check for MAIN_DRIVE in path before creating directory
                    if 'MAIN_DRIVE' in dest_file.parts:
                        error_msg = (f"CRITICAL BUG: Attempting to create directory with MAIN_DRIVE: {dest_file}. "
                                   f"Path parts: {dest_file.parts}")
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if file_a.is_symlink():
                            link_target = os.readlink(file_a)
                            if dest_file.exists() or dest_file.is_symlink():
                                dest_file.unlink()
                            dest_file.symlink_to(link_target)
                            self.logger.debug(f"Copied symlink: {file_a} -> {dest_file}")
                        else:
                            shutil.copy2(file_a, dest_file)
                            # Use shortened paths for compact output
                            # Source includes filename, destination shows directory only
                            short_src = self._shorten_path(file_a, base_paths, include_filename=True)
                            short_dest = self._shorten_path(dest_file, base_paths, include_filename=False)
                            self.logger.info(f"{short_src} → {short_dest}")
                        copied_a_to_b += 1
                    except Exception as e:
                        self.logger.error(f"Failed to copy {file_a} to {dest_file}: {e}")
            
            if copied_a_to_b > 0 or copied_b_to_a > 0:
                short_a = self._shorten_path(folder_a, base_paths, include_filename=False)
                short_b = self._shorten_path(folder_b, base_paths, include_filename=False)
                self.logger.info(f"Sync complete: {copied_a_to_b} files A→B, {copied_b_to_a} files B→A ({short_a} ↔ {short_b})")
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
                    
                    # CRITICAL: Check for MAIN_DRIVE in path before creating directory
                    if 'MAIN_DRIVE' in target_item.parts:
                        error_msg = (f"CRITICAL BUG: Attempting to create directory with MAIN_DRIVE: {target_item}. "
                                   f"Path parts: {target_item.parts}")
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    
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
    
    # Name ending with " (1)" or " (1).ext" -> destination should use "_1" / "_1.ext" to match dedupe_mac_copies
    _SUFFIX_1_PATTERN = re.compile(r"^(.+?) \(1\)(\..*)?$", re.IGNORECASE)
    
    def _normalize_dedupe_name(self, name: str) -> Optional[str]:
        """If name is 'X (1)' or 'X (1).ext', return 'X_1' or 'X_1.ext'; else None."""
        m = self._SUFFIX_1_PATTERN.match(name)
        if not m:
            return None
        base, ext = m.group(1), m.group(2) or ""
        return f"{base}_1{ext}"
    
    def _resolve_dedupe_target(self, source: Path, dest_dir: Path, is_dir: bool = False) -> Tuple[Optional[Path], bool]:
        """
        When source name ends with ' (1)' or ' (1).ext', resolve the destination path so we never create
        that name in the destination. Return (effective_target_path, skip).
        - skip=True: destination already has this content (e.g. x.png or x_1.png with same content); do not copy.
        - skip=False: copy to effective_target_path (normalized name, or _2, _3 if _1 exists with different content).
        """
        norm = self._normalize_dedupe_name(source.name)
        if not norm:
            return None, False
        # Candidates that might already hold this content (destination may have been deduped)
        base, ext = (source.stem.rstrip(" (1)"), source.suffix) if source.is_file() else (Path(norm).stem.replace("_1", ""), "")
        if is_dir:
            candidates = [dest_dir / norm]
            preferred = dest_dir / norm
        else:
            # For files: dest might have x.png (merged) or x_1.png
            base_only = base.replace(" (1)", "").strip()
            candidates = [dest_dir / f"{base_only}{ext}", dest_dir / norm]
            preferred = dest_dir / norm
        for c in candidates:
            if c.exists() and c.is_file() and source.is_file():
                try:
                    if c.stat().st_size == source.stat().st_size and self._get_file_hash(source) == self._get_file_hash(c):
                        return None, True
                except Exception:
                    pass
        # Choose destination name: preferred if it doesn't exist, else _2, _3, ...
        if is_dir:
            stem, suffix = norm, ""
            if preferred.exists():
                i = 2
                while (dest_dir / f"{Path(norm).stem.rsplit('_', 1)[0]}_{i}{suffix}").exists():
                    i += 1
                norm = f"{Path(norm).stem.rsplit('_', 1)[0]}_{i}{suffix}"
            return dest_dir / norm, False
        stem, ext = Path(norm).stem, Path(norm).suffix
        base_stem = stem.rsplit("_", 1)[0] if "_" in stem else stem
        p = dest_dir / norm
        if not p.exists():
            return p, False
        i = 2
        while (dest_dir / f"{base_stem}_{i}{ext}").exists():
            i += 1
        return dest_dir / f"{base_stem}_{i}{ext}", False
    
    def copy_with_symlink_preservation(self, source: Path, target: Path):
        """Copy file or directory preserving symlinks. Never creates 'X (1)' in destination; uses normalized names."""
        # Resolve destination when source name ends with " (1)" so we don't create that name in destination
        if self._SUFFIX_1_PATTERN.match(source.name):
            dest_dir = target.parent
            effective, skip = self._resolve_dedupe_target(source, dest_dir, is_dir=source.is_dir())
            if skip:
                self.logger.debug(f"Skipping copy (content already present): {source} -> {dest_dir}")
                return
            if effective is not None:
                target = effective
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
            
            # CRITICAL: Check for MAIN_DRIVE in backup target path
            if 'MAIN_DRIVE' in target_path.parts:
                error_msg = (f"CRITICAL BUG: Backup target path contains MAIN_DRIVE: {target_path}. "
                           f"Source: {source_path}, Path parts: {target_path.parts}")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Never create "X (1)" in destination; use normalized name (x_1, x_1.ext) to match dedupe_mac_copies
            if self._SUFFIX_1_PATTERN.match(source_path.name):
                effective, skip = self._resolve_dedupe_target(source_path, target_path.parent, is_dir=source_path.is_dir())
                if skip:
                    self.logger.debug(f"Skipping backup (content already present): {source_path}")
                    return True
                if effective is not None:
                    target_path = effective
            
            if source_path.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self.copy_with_symlink_preservation(source_path, target_path)
            elif source_path.is_dir():
                for item in source_path.rglob('*'):
                    if item.is_file():
                        rel_item = item.relative_to(source_path)
                        target_item = target_path / rel_item
                        # CRITICAL: Check for MAIN_DRIVE in backup target path
                        if 'MAIN_DRIVE' in target_item.parts:
                            error_msg = (f"CRITICAL BUG: Backup target path contains MAIN_DRIVE: {target_item}. "
                                       f"Source: {item}, Path parts: {target_item.parts}")
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
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
    
    def __init__(self, config_file: str = "config.yaml", test_mode: bool = False):
        self.config_file = config_file
        self.test_mode = test_mode
        self.config = self._load_config()
        self.running = True  # Always set to True initially (even for --scan-once)
        self.logger = self._setup_logging()
        
        # CRITICAL: Check if MAIN_DRIVE directory exists in current working directory (indicates previous bug)
        cwd = Path.cwd()
        main_drive_dir = cwd / 'MAIN_DRIVE'
        if main_drive_dir.exists() and main_drive_dir.is_dir():
            error_msg = (f"CRITICAL: Found incorrectly created 'MAIN_DRIVE' directory at: {main_drive_dir}\n"
                        f"This indicates a previous bug where drive placeholders were not properly resolved.\n"
                        f"You MUST delete this directory before running the organizer again, or it will cause data corruption.\n"
                        f"To fix:\n"
                        f"  1. Stop the file organizer\n"
                        f"  2. Review the contents of {main_drive_dir} to see what was incorrectly copied\n"
                        f"  3. Delete the directory: rm -rf {main_drive_dir}\n"
                        f"  4. Check your config.yaml to ensure all drive placeholders are properly resolved")
            self.logger.error(error_msg)
            if not test_mode:
                print("\n" + "=" * 70)
                print("CRITICAL ERROR: Incorrectly created MAIN_DRIVE directory detected!")
                print("=" * 70)
                print(error_msg)
                print("=" * 70 + "\n")
                raise ValueError("MAIN_DRIVE directory exists - this indicates a configuration error. Please fix before continuing.")
        
        # PRODUCTION MODE: Strict validation (exit on any issues)
        if not test_mode:
            self._validate_config_structure()
            self._check_unconfigured_drives()
            self._resolve_drive_placeholders()
            self._filter_unavailable_drives()  # Drop pairs for drives not present on this machine
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
                template_file = "config_template.yaml"
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
            # Create a test source_folders list for backward compatibility (used by file scanning)
            if 'source_folders' not in self.config:
                self.config['source_folders'] = []
            self.config['source_folders'] = [str(Path.cwd() / 'test')]
            if 'exclude_folders' not in self.config:
                self.config['exclude_folders'] = []
            self.config['exclude_folders'] = [str(Path.cwd() / 'test' / 'organized')]
            self.config['output_base'] = str(Path.cwd() / 'test' / 'organized')
            self.config['enable_duplicate_detection'] = False
            self.config['enable_folder_sync'] = False
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
        self.folder_sync = FolderSynchronizer(self.logger, self.config)
        self.background_backup = BackgroundBackup(self.logger, self.config)
        self.content_analyzer = DynamicContentAnalyzer(self.logger, self.config)
        self.semantic_categorizer = SemanticCategorizer(self.logger, self.config)
        
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
        
        # State cache — load first so _index_existing_organized_links can use it
        # to skip the filesystem walk on warm starts.
        if self.test_mode:
            self._state_cache_file = Path.cwd() / '.file_organizer_state.json'
        else:
            self._state_cache_file = Path.home() / '.file_organizer_state.json'
        self._state_cache: Dict[str, list] = {}
        self._load_state_cache()

        # Index existing symlinks in output_base so file_link_counts is
        # accurate even for files we skip on this run.
        self._index_existing_organized_links()

        # Track processed files
        self.processed_files = set()
        self._newly_processed_files = set()  # files analyzed on THIS run (not skipped)

        # Files examined counter for periodic progress logging — reset each cycle
        self._scan_file_count: int = 0

        # Per-cycle stats — reset at the start of each run_full_cycle()
        self._cycle_stats: Dict[str, int] = {
            'gate1_skipped': 0,   # had links already, skipped
            'gate2_skipped': 0,   # unchanged + no links, skipped
            'gate3_updated': 0,   # file changed, re-analyzed
            'new_files': 0,       # not in cache, analyzed fresh
            'links_created': 0,   # symlinks created this cycle
            'files_pruned': 0,    # deleted source files pruned from cache
            'dirs_cleaned': 0,    # empty dirs removed after pruning
        }
    
    def _get_organised_base(self) -> Path:
        """Get the organised_base path from config, with default fallback."""
        backup_base = self.config.get('softlink_backup_base', '~/organised')
        # Expand ~ to home directory
        if backup_base.startswith('~'):
            backup_base = str(Path.home()) + backup_base[1:]
        return Path(backup_base)
    
    def _index_existing_organized_links(self):
        """Populate file_link_counts from state cache (fast) or filesystem walk (cold start).

        The state cache records every link created for each source file as
        [mtime, size, [rel_link_1, ...]]. When the cache is warm we skip the
        os.walk() entirely — just one lstat() per file to verify the first
        cached link still exists (self-healing if output_base is cleared).
        """
        output_base = Path(self.config['output_base'])
        if not output_base.exists():
            return

        # Fast path: derive counts from state cache (O(n) dict iteration, 1 lstat per file).
        # We verify the first cached link actually exists on disk so that an externally
        # cleared output_base doesn't leave Gate 1 permanently blocking re-analysis.
        if self._state_cache:
            count = 0
            for file_key, cached in self._state_cache.items():
                link_rel_paths = cached[2] if len(cached) > 2 else []
                if not link_rel_paths:
                    continue
                # One lstat+readlink per file — verify the symlink exists and points to
                # this file (not a collision victim where another file stole the slot).
                first_link = output_base / link_rel_paths[0]
                try:
                    if not first_link.is_symlink():
                        continue  # output_base was cleared externally
                    if str(first_link.resolve()) != file_key:
                        continue  # Link was overwritten by a different source file
                except (OSError, ValueError):
                    continue
                self.file_link_counts[file_key] = len(link_rel_paths)
                count += len(link_rel_paths)
            if count:
                self.logger.info(
                    f"Indexed {count} existing symlinks across "
                    f"{len(self.file_link_counts)} files from state cache"
                )
            return

        # Cold start (no cache yet): walk the filesystem as before.
        count = 0
        for root, _dirs, files in os.walk(output_base):
            for name in files:
                link_path = Path(root) / name
                if link_path.is_symlink():
                    try:
                        target = link_path.resolve()
                        if target.exists():
                            self.file_link_counts[str(target)] += 1
                            count += 1
                    except (OSError, ValueError):
                        pass

        if count:
            self.logger.info(
                f"Indexed {count} existing symlinks across "
                f"{len(self.file_link_counts)} files in {output_base}"
            )

    def _cleanup_symlink_tree(self, base: Path, remove_excluded_targets: bool = False):
        """Remove broken symlinks (and optionally symlinks to excluded paths) under base; prune empty dirs.
        Returns (removed, broken, excluded)."""
        removed = broken = excluded = 0
        if not base.exists():
            return removed, broken, excluded
        for root, dirs, files in os.walk(base, topdown=False):
            for name in files:
                link_path = Path(root) / name
                if not link_path.is_symlink():
                    continue
                try:
                    target = link_path.resolve()
                    if not target.exists():
                        link_path.unlink()
                        broken += 1
                        removed += 1
                    elif remove_excluded_targets and self.should_exclude_path(target):
                        link_path.unlink()
                        excluded += 1
                        removed += 1
                except Exception:
                    pass
            dir_path = Path(root)
            if dir_path != base:
                try:
                    if not list(dir_path.iterdir()):
                        dir_path.rmdir()
                except Exception:
                    pass
        return removed, broken, excluded

    def cleanup_organized(self):
        """Clean both output_base (~/organized) and softlink_backup_base (~/organised).
        - ~/organized: remove broken symlinks, symlinks to excluded paths, and empty dirs.
        - ~/organised: remove broken symlinks and empty dirs (backup contents stay)."""
        ob = self.config.get('output_base', '')
        if ob.startswith('~'):
            ob = str(Path.home()) + ob[1:]
        output_base = Path(ob)

        organised_base = self._get_organised_base()

        total_removed = total_broken = total_excluded = 0

        # Clean ~/organized (category symlinks)
        if output_base.exists():
            rem, brk, exc = self._cleanup_symlink_tree(output_base, remove_excluded_targets=True)
            total_removed += rem
            total_broken += brk
            total_excluded += exc
            self.logger.info(
                f"Cleanup {output_base}: removed {rem} symlinks ({brk} broken, {exc} excluded)"
            )
            print(f"~/organized: removed {rem} symlinks ({brk} broken, {exc} pointing to excluded paths)")
        else:
            self.logger.info("Nothing to clean in output_base — does not exist")
            print("~/organized: (not present, skipped)")

        # Clean ~/organised (softlink backup: .git, __pycache__, etc.)
        if organised_base.exists():
            rem, brk, exc = self._cleanup_symlink_tree(organised_base, remove_excluded_targets=False)
            total_removed += rem
            total_broken += brk
            self.logger.info(
                f"Cleanup {organised_base}: removed {rem} symlinks ({brk} broken)"
            )
            print(f"~/organised: removed {rem} symlinks ({brk} broken)")
        else:
            self.logger.info("Nothing to clean in softlink_backup_base — does not exist")
            print("~/organised: (not present, skipped)")

        self.logger.info(
            f"Cleanup complete: {total_removed} symlinks removed total "
            f"({total_broken} broken, {total_excluded} excluded)"
        )
        print(f"Total: {total_removed} symlinks removed ({total_broken} broken, {total_excluded} excluded)")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_file):
            # Config file doesn't exist - require it even in test mode
            print("\n" + "=" * 70)
            print("ERROR: Configuration file not found!")
            print("=" * 70)
            print(f"\nThe file '{self.config_file}' does not exist.")
            print("\nA valid configuration file is required, even in test mode.")
            
            # Check if template exists
            template_file = "config_template.yaml"
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
        
        # Check for TAB characters in the file (YAML doesn't allow tabs)
        try:
            with open(self.config_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    if '\t' in line:
                        print("\n" + "=" * 70)
                        print("INVALID YAML CONFIGURATION")
                        print("=" * 70)
                        print(f"\nThe file '{self.config_file}' contains TAB characters on line {line_num}.")
                        print("\nYAML does not allow TAB characters for indentation. You must use SPACES instead.")
                        print("\nThis is a common mistake when editing YAML files.")
                        print("\nTo fix this:")
                        print("  1. Open the file in your editor")
                        print(f"  2. Find line {line_num} and replace any TAB characters with spaces")
                        print("  3. Most editors have a 'Show Whitespace' or 'Show Invisibles' option")
                        print("  4. You can also use: sed -i '' 's/\\t/  /g' " + self.config_file)
                        print("\n" + "=" * 70 + "\n")
                        sys.exit(1)
        except Exception as e:
            print("\n" + "=" * 70)
            print("ERROR: Could not read configuration file")
            print("=" * 70)
            print(f"\nFailed to read '{self.config_file}':")
            print(f"  {e}")
            print("\n" + "=" * 70 + "\n")
            sys.exit(1)
        
        # Try to load and parse the YAML file
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                print("\n" + "=" * 70)
                print("INVALID YAML CONFIGURATION")
                print("=" * 70)
                print(f"\nThe file '{self.config_file}' appears to be empty or contains no valid YAML.")
                print("\nPlease ensure the file contains valid YAML configuration.")
                print("\n" + "=" * 70 + "\n")
                sys.exit(1)
            
            # Always validate config if it exists (both test and production modes)
            self._validate_config_sanity(config)
            return config
            
        except yaml.YAMLError as e:
            # YAML syntax error - provide user-friendly error messages
            print("\n" + "=" * 70)
            print("INVALID YAML CONFIGURATION")
            print("=" * 70)
            
            error_msg = str(e)
            if hasattr(e, 'problem_mark'):
                mark = e.problem_mark
                line_num = mark.line + 1
                col_num = mark.column + 1
                print(f"\nThe file '{self.config_file}' contains invalid YAML at line {line_num}, column {col_num}:")
                print(f"  {error_msg}")
                
                # Check for common indentation issues
                if 'indentation' in error_msg.lower() or 'expected' in error_msg.lower():
                    print("\nThis looks like an indentation error. YAML is very sensitive to indentation.")
                    print("\nCommon causes:")
                    print("  - Mixing spaces and tabs (YAML requires spaces only)")
                    print("  - Incorrect number of spaces for indentation")
                    print("  - Inconsistent indentation levels")
                    print("\nTo fix this:")
                    print(f"  1. Check line {line_num} in your config file")
                    print("  2. Ensure you're using SPACES (not tabs) for indentation")
                    print("  3. YAML typically uses 2 spaces per indentation level")
                    print("  4. Make sure all items at the same level use the same indentation")
                else:
                    print("\nThis is a YAML syntax error.")
                    print("\nTo fix this:")
                    print("  1. Check the line and column mentioned above")
                    print("  2. Ensure proper YAML syntax (colons, dashes, quotes, etc.)")
                    print("  3. Use a YAML validator: https://www.yamllint.com/")
            else:
                print(f"\nThe file '{self.config_file}' contains invalid YAML:")
                print(f"  {error_msg}")
                print("\nThis is a common mistake when editing YAML configuration files.")
                print("\nTo fix this:")
                print("  1. Check for syntax errors (missing colons, incorrect indentation, etc.)")
                print("  2. Use a YAML validator: https://www.yamllint.com/")
                print("  3. Or start fresh: cp config_template.yaml " + self.config_file)
            
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
    
    def _validate_config_sanity(self, config: Dict):
        """Validate config for basic sanity (both test and production modes)."""
        errors = []
        warnings = []
        
        # Check for circular references in sync_pairs
        if 'sync_pairs' in config and isinstance(config['sync_pairs'], list):
            drives = config.get('drives', {})
            for i, pair in enumerate(config['sync_pairs']):
                if not isinstance(pair, dict):
                    continue
                
                # Support both new format (folders) and old format (source/target)
                if 'folders' in pair:
                    # New format
                    folders = pair['folders']
                    if not isinstance(folders, list) or len(folders) != 2:
                        errors.append(f"sync_pairs[{i}]: 'folders' must be a list of exactly 2 paths")
                        continue
                    path1 = folders[0]
                    path2 = folders[1]
                elif 'source' in pair and 'target' in pair:
                    # Old format (backward compatibility)
                    path1 = pair['source']
                    path2 = pair['target']
                else:
                    # Comment-only entry, skip
                    continue
                
                # Resolve drive placeholders first, then expand and resolve paths
                try:
                    path1_resolved_str = self._resolve_path_with_drives(path1, drives)
                    path2_resolved_str = self._resolve_path_with_drives(path2, drives)
                    path1_resolved = str(Path(path1_resolved_str).expanduser().resolve())
                    path2_resolved = str(Path(path2_resolved_str).expanduser().resolve())
                except Exception as e:
                    errors.append(f"sync_pairs[{i}]: Invalid path - {e}")
                    continue
                
                # Check if paths are the same
                if path1_resolved == path2_resolved:
                    errors.append(f"sync_pairs[{i}]: Folders are identical: '{path1}' → '{path1_resolved}'")
                
                # Check if one is a subdirectory of the other (potential circular reference)
                if path1_resolved.startswith(path2_resolved + '/') or path2_resolved.startswith(path1_resolved + '/'):
                    warnings.append(f"sync_pairs[{i}]: potential circular reference - folders are nested")

        # Check for circular references in one_way_pairs
        if 'one_way_pairs' in config and isinstance(config['one_way_pairs'], list):
            drives = config.get('drives', {})
            for i, pair in enumerate(config['one_way_pairs']):
                if not isinstance(pair, dict):
                    continue
                
                # Support both new format (folders) and old format (source/target)
                if 'folders' in pair:
                    # New format
                    folders = pair['folders']
                    if not isinstance(folders, list) or len(folders) != 2:
                        errors.append(f"one_way_pairs[{i}]: 'folders' must be a list of exactly 2 paths")
                        continue
                    path1 = folders[0]
                    path2 = folders[1]
                elif 'source' in pair and 'target' in pair:
                    # Old format (backward compatibility)
                    path1 = pair['source']
                    path2 = pair['target']
                else:
                    # Comment-only entry, skip
                    continue
                
                # Resolve drive placeholders first, then expand and resolve paths
                try:
                    path1_resolved_str = self._resolve_path_with_drives(path1, drives)
                    path2_resolved_str = self._resolve_path_with_drives(path2, drives)
                    path1_resolved = str(Path(path1_resolved_str).expanduser().resolve())
                    path2_resolved = str(Path(path2_resolved_str).expanduser().resolve())
                except Exception as e:
                    errors.append(f"one_way_pairs[{i}]: Invalid path - {e}")
                    continue
                
                # Check if paths are the same
                if path1_resolved == path2_resolved:
                    errors.append(f"one_way_pairs[{i}]: Folders are identical: '{path1}' → '{path1_resolved}'")
                
                # Check if one is a subdirectory of the other (potential circular reference)
                if path1_resolved.startswith(path2_resolved + '/') or path2_resolved.startswith(path1_resolved + '/'):
                    warnings.append(f"one_way_pairs[{i}]: potential circular reference - folders are nested")
        
        # Check for output_base conflicts (if output_base exists)
        if 'output_base' in config:
            try:
                output_resolved = str(Path(config['output_base']).expanduser().resolve())
                # Check if output_base is inside any sync pair folders
                if 'sync_pairs' in config and isinstance(config['sync_pairs'], list):
                    for i, pair in enumerate(config['sync_pairs']):
                        if not isinstance(pair, dict):
                            continue
                        try:
                            if 'folders' in pair:
                                folders = pair['folders']
                                if isinstance(folders, list) and len(folders) == 2:
                                    path1_resolved = str(Path(folders[0]).expanduser().resolve())
                                    path2_resolved = str(Path(folders[1]).expanduser().resolve())
                                    if output_resolved.startswith(path1_resolved + '/') or output_resolved.startswith(path2_resolved + '/'):
                                        warnings.append(f"output_base '{config['output_base']}' is inside sync_pairs[{i}] - this may cause issues")
                            elif 'source' in pair and 'target' in pair:
                                # Old format
                                source_resolved = str(Path(pair['source']).expanduser().resolve())
                                target_resolved = str(Path(pair['target']).expanduser().resolve())
                                if output_resolved.startswith(source_resolved + '/') or output_resolved.startswith(target_resolved + '/'):
                                    warnings.append(f"output_base '{config['output_base']}' is inside sync_pairs[{i}] - this may cause issues")
                        except Exception:
                            pass
            except Exception:
                pass
        
        # Check if output_base is inside any one-way pair folders
        if 'output_base' in config:
            try:
                output_resolved = str(Path(config['output_base']).expanduser().resolve())
                if 'one_way_pairs' in config and isinstance(config['one_way_pairs'], list):
                    for i, pair in enumerate(config['one_way_pairs']):
                        if not isinstance(pair, dict):
                            continue
                        try:
                            if 'folders' in pair:
                                folders = pair['folders']
                                if isinstance(folders, list) and len(folders) == 2:
                                    path1_resolved = str(Path(folders[0]).expanduser().resolve())
                                    path2_resolved = str(Path(folders[1]).expanduser().resolve())
                                    if output_resolved.startswith(path1_resolved + '/') or output_resolved.startswith(path2_resolved + '/'):
                                        warnings.append(f"output_base '{config['output_base']}' is inside one_way_pairs[{i}] - this may cause issues")
                            elif 'source' in pair and 'target' in pair:
                                # Old format
                                source_resolved = str(Path(pair['source']).expanduser().resolve())
                                target_resolved = str(Path(pair['target']).expanduser().resolve())
                                if output_resolved.startswith(source_resolved + '/') or output_resolved.startswith(target_resolved + '/'):
                                    warnings.append(f"output_base '{config['output_base']}' is inside one_way_pairs[{i}] - this may cause issues")
                        except Exception:
                            pass
            except Exception:
                pass
        
        # Check for unknown top-level config keys (catches typos)
        KNOWN_KEYS = {
            'drives', 'sync_pairs', 'one_way_pairs', 'source_folders',
            'softlink_folder_patterns', 'empty_folder_patterns',
            'exclude_patterns', 'exclude_folders', 'exclude_extensions',
            'output_base', 'softlink_backup_base',
            'min_file_size', 'max_file_size',
            'enable_content_analysis', 'min_image_pixels_for_ocr', 'min_video_size_for_ocr',
            'enable_duplicate_detection', 'dedupe_dry_run',
            'enable_folder_sync',
            'scan_interval', 'flaky_volume_retries', 'retry_delay',
            'use_rsync', 'rsync_checksum_mode', 'rsync_size_only',
            'rsync_additional_args', 'rsync_disable_mmap',
            'max_drive_usage_percent', 'sync_chunk_subfolders',
            'sync_chunk_concurrency', 'sync_timeout_minutes',
            'max_soft_links_per_file',
            'enable_semantic_categories', 'semantic_confidence_threshold',
            'ml_content_analysis',
            'enable_background_backup', 'backup_drive_path', 'backup_directories',
        }
        unknown = [k for k in config if k not in KNOWN_KEYS]
        if unknown:
            warnings.append(
                f"Unknown config key(s): {', '.join(sorted(unknown))} — possible typo?"
            )

        # Check file size constraint
        min_sz = config.get('min_file_size')
        max_sz = config.get('max_file_size')
        if isinstance(min_sz, (int, float)) and isinstance(max_sz, (int, float)):
            if min_sz > max_sz:
                errors.append(
                    f"min_file_size ({min_sz}) must be ≤ max_file_size ({max_sz})"
                )

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
        rsync_disable_mmap_default = sys.platform == 'darwin'
        defaults = {
            'flaky_volume_retries': 3,
            'retry_delay': 5,
            'scan_interval': 3600,
            'max_file_size': 104857600,
            'enable_content_analysis': True,
            'enable_duplicate_detection': False,
            'enable_folder_sync': False,
            'enable_background_backup': False,
            'use_rsync': True,
            'rsync_checksum_mode': 'timestamp',
            'rsync_size_only': False,
            'rsync_additional_args': [],
            'rsync_disable_mmap': rsync_disable_mmap_default,
            'sync_chunk_subfolders': 30,
            'sync_chunk_concurrency': 1,
            'sync_timeout_minutes': 60,
            'max_drive_usage_percent': 90,
            'max_soft_links_per_file': 6,
            'min_file_size': 1024,
            'softlink_folder_patterns': [
                '.git', '.hg', '.svn', '.cvs', '__pycache__', '.pytest_cache',
                '.mypy_cache', '.tox', '.venv', 'venv', 'env'
            ],
            'empty_folder_patterns': [
                'node_modules', '_build', 'deps', 'ebin', 'dist', 'build',
                'target', '.next', '.nuxt', '.cache', '.parcel-cache',
                'coverage', '.nyc_output', 'elm-stuff', '.elixir_ls',
                '.stack-work'
            ],
            'exclude_patterns': [
                '.DS_Store', '*.pyc', '*.log', '.Spotlight-V100',
                '.TemporaryItems', '.fseventsd', '.DocumentRevisions-V100',
                '.Trash', '.Trashes', '*_files',
            ],
            'exclude_extensions': ['.beam', '.pyc', '.o', '.so'],
            'exclude_folders': [],
            'source_folders': [],
            'sync_pairs': [],
            'one_way_pairs': [],
            'backup_directories': [],
            'drives': {},
            'output_base': '',
            'backup_drive_path': '',
            'enable_semantic_categories': True,
            'semantic_confidence_threshold': 0.35,
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
    
    def _resolve_path_with_drives(self, path: str, drives: Dict, max_depth: int = 10) -> str:
        """
        Recursively resolve a path containing drive placeholders.
        Handles nested drive references like PROTON_DRIVE: "MAIN_DRIVE/ProtonDrive"
        
        Simple approach: Replace drive names with their values until no more replacements occur.
        """
        if max_depth <= 0:
            # Prevent infinite recursion
            self.logger.warning(f"Max recursion depth reached resolving path: {path}")
            return str(Path(path).expanduser())
        
        # If path is already absolute and doesn't contain any drive placeholders, return it
        if path.startswith('/') or path.startswith('~'):
            # Check if it still contains any drive placeholders (shouldn't happen, but be safe)
            for drive_name in drives.keys():
                if drive_name.startswith('comment'):
                    continue
                if drive_name in path:
                    # Still has a drive placeholder in an absolute path - this is an error
                    error_msg = (f"CRITICAL BUG: Absolute path '{path}' still contains drive placeholder '{drive_name}'. "
                               f"This should not happen after resolution.")
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            return str(Path(path).expanduser())
        
        # Path is relative - check if it starts with a drive name
        original_path = path
        for drive_name, drive_path in drives.items():
            # Skip comment keys
            if drive_name.startswith('comment'):
                continue
            
            # Check if path starts with drive name followed by '/' or is exactly the drive name
            if path.startswith(drive_name + '/') or path == drive_name:
                # Replace the drive name with its path (only the first occurrence)
                if path == drive_name:
                    # Exact match - replace entire path
                    resolved = drive_path
                else:
                    # Path starts with drive_name/ - replace drive_name with drive_path
                    resolved = drive_path + path[len(drive_name):]
                
                # Recursively resolve any remaining drive placeholders in the resolved path
                resolved = self._resolve_path_with_drives(resolved, drives, max_depth - 1)
                # Expand ~ if present and return
                return str(Path(resolved).expanduser())
        
        # No drive placeholder found at the start - check if path looks like it contains an unresolved drive placeholder
        path_parts = path.split('/')
        if path_parts and path_parts[0]:
            first_part = path_parts[0]
            # Check if first part looks like a drive placeholder
            if not path.startswith('/') and not path.startswith('.') and not path.startswith('~'):
                if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                   first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                    # This looks like an unresolved drive placeholder!
                    if not self.test_mode:
                        if drives:
                            if first_part not in drives:
                                error_msg = (f"CRITICAL BUG: Unresolved drive placeholder '{first_part}' in path '{path}'. "
                                           f"Available drives: {list(drives.keys())}. "
                                           f"This will create a literal '{first_part}' directory. "
                                           f"Please ensure '{first_part}' is defined in the 'drives' section.")
                                self.logger.error(error_msg)
                                raise ValueError(error_msg)
                        else:
                            error_msg = (f"CRITICAL BUG: Path '{path}' contains drive placeholder '{first_part}' but no 'drives' section found. "
                                       f"This will create a literal '{first_part}' directory.")
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                    elif drives and first_part not in drives:
                        self.logger.warning(f"Unresolved drive placeholder '{first_part}' in path '{path}' - treating as relative path")
        
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
        required_keys = ['output_base']
        for key in required_keys:
            if key not in self.config:
                errors.append(f"Missing required key: '{key}'")
        
        # Validate sync_pairs (required for sync functionality)
        if 'sync_pairs' in self.config:
            if not isinstance(self.config['sync_pairs'], list):
                errors.append("'sync_pairs' must be a list")
        
        # Validate one_way_pairs (optional one-way sync)
        if 'one_way_pairs' in self.config:
            if not isinstance(self.config['one_way_pairs'], list):
                errors.append("'one_way_pairs' must be a list")
        
        # Warn if sync is enabled but no pairs are configured
        if self.config.get('enable_folder_sync', False):
            sync_pairs = self.config.get('sync_pairs', [])
            one_way_pairs = self.config.get('one_way_pairs', [])
            if len(sync_pairs) == 0 and len(one_way_pairs) == 0:
                warnings.append("'sync_pairs' and 'one_way_pairs' are empty but 'enable_folder_sync' is true - no folders will be synced")
        
        # Legacy support: validate source_folders if present (for backward compatibility)
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
                    
                    # Support both new format (folders) and old format (source/target)
                    if 'folders' in pair:
                        # New format: folders must be a list of exactly 2 paths
                        if not isinstance(pair['folders'], list):
                            errors.append(f"sync_pairs[{i}]: 'folders' must be a list")
                        elif len(pair['folders']) != 2:
                            errors.append(f"sync_pairs[{i}]: 'folders' must contain exactly 2 paths (has {len(pair['folders'])})")
                    elif 'source' in pair and 'target' in pair:
                        # Old format (backward compatibility) - both required
                        pass
                    else:
                        # Must have either 'folders' or both 'source' and 'target'
                        errors.append(f"sync_pairs[{i}] must have either 'folders' (list of 2 paths) or 'source'/'target' (has: {list(pair.keys())})")
        
        # Validate one_way_pairs structure
        if 'one_way_pairs' in self.config:
            if not isinstance(self.config['one_way_pairs'], list):
                errors.append("'one_way_pairs' must be a list")
            else:
                for i, pair in enumerate(self.config['one_way_pairs']):
                    if not isinstance(pair, dict):
                        errors.append(f"one_way_pairs[{i}] must be a dictionary")
                        continue
                    
                    # Check if it's just a comment entry (only has 'comment' keys)
                    keys = [k for k in pair.keys() if not k.startswith('comment')]
                    if len(keys) == 0:
                        # Just a comment, skip validation
                        continue
                    
                    # Support both new format (folders) and old format (source/target)
                    if 'folders' in pair:
                        # New format: folders must be a list of exactly 2 paths
                        if not isinstance(pair['folders'], list):
                            errors.append(f"one_way_pairs[{i}]: 'folders' must be a list")
                        elif len(pair['folders']) != 2:
                            errors.append(f"one_way_pairs[{i}]: 'folders' must contain exactly 2 paths (has {len(pair['folders'])})")
                    elif 'source' in pair and 'target' in pair:
                        # Old format (backward compatibility) - both required
                        pass
                    else:
                        # Must have either 'folders' or both 'source' and 'target'
                        errors.append(f"one_way_pairs[{i}] must have either 'folders' (list of 2 paths) or 'source'/'target' (has: {list(pair.keys())})")
        
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
            print("See config_template.yaml for a valid example.")
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
            print("\nSee config_template.yaml for an example.")
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
        # FIRST: Resolve drive definitions themselves (in case they reference other drives)
        # This handles cases like PROTON_DRIVE: "MAIN_DRIVE/ProtonDrive"
        # Use iterative resolution to handle interdependent drives
        if 'drives' in self.config:
            drives = self.config['drives'].copy()  # Work with a copy
            resolved_drives = {}
            max_iterations = 10  # Prevent infinite loops
            
            # Initialize resolved_drives with all drives (will resolve iteratively)
            for drive_name, drive_path in drives.items():
                if drive_name.startswith('comment'):
                    resolved_drives[drive_name] = drive_path
                elif isinstance(drive_path, str):
                    resolved_drives[drive_name] = drive_path
                else:
                    resolved_drives[drive_name] = drive_path
            
            # Iterative resolution: keep resolving until no more changes
            converged = False
            for iteration in range(max_iterations):
                changed = False
                for drive_name, drive_path in resolved_drives.items():
                    if drive_name.startswith('comment'):
                        continue
                    if isinstance(drive_path, str):
                        # Try to resolve using current resolved_drives
                        new_path = self._resolve_path_with_drives(drive_path, resolved_drives)
                        # Only update if path actually changed (resolved a placeholder)
                        if new_path != drive_path:
                            resolved_drives[drive_name] = new_path
                            changed = True
                
                if not changed:
                    converged = True
                    break  # All drives resolved
            
            if not converged:
                self.logger.warning("Drive resolution did not converge after max iterations - some drives may not be fully resolved")
            
            # Update drives with resolved values
            self.config['drives'] = resolved_drives
        
        # NOW resolve paths using the resolved drives
        drives = self.config.get('drives', {})
        
        # Resolve drive placeholders in config (only if they exist - they're optional now)
        if 'source_folders' in self.config:
            self._resolve_placeholders('source_folders')
        if 'exclude_folders' in self.config:
            self._resolve_placeholders('exclude_folders')
        self._resolve_placeholder_value('output_base')
        if 'backup_drive_path' in self.config:
            self._resolve_placeholder_value('backup_drive_path')
        
        # Resolve drive placeholders in sync_pairs
        # This is CRITICAL - all drive placeholders MUST be replaced with actual paths here
        if 'sync_pairs' in self.config:
            self.logger.info(f"Resolving drive placeholders in {len(self.config['sync_pairs'])} sync pairs...")
            for i, pair in enumerate(self.config['sync_pairs']):
                if 'folders' in pair:
                    # New format: resolve each folder in the list
                    if isinstance(pair['folders'], list):
                        resolved_folders = []
                        for folder in pair['folders']:
                            self.logger.debug(f"Resolving path '{folder}' in sync_pairs[{i}]")
                            resolved = self._resolve_path_with_drives(folder, drives)
                            self.logger.debug(f"  Resolved to: '{resolved}'")
                            
                            # CRITICAL: Validate that resolution actually worked
                            # The resolved path MUST be absolute and MUST NOT contain any drive placeholder strings
                            if not resolved.startswith('/') and not resolved.startswith('~'):
                                first_part = resolved.split('/')[0] if '/' in resolved else resolved
                                if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                                   first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                                    if first_part not in drives:
                                        error_msg = (f"CRITICAL BUG: Path resolution failed for '{folder}' in sync_pairs[{i}]. "
                                                   f"Resolved to '{resolved}' which still contains unresolved drive placeholder '{first_part}'. "
                                                   f"Available drives: {list(drives.keys())}. "
                                                   f"This will create a literal '{first_part}' directory. "
                                                   f"Please check your drives configuration.")
                                        self.logger.error(error_msg)
                                        raise ValueError(error_msg)
                            
                            # Additional check: resolved path must not contain "MAIN_DRIVE" anywhere
                            if 'MAIN_DRIVE' in resolved:
                                error_msg = (f"CRITICAL BUG: Resolved path '{resolved}' still contains 'MAIN_DRIVE' string. "
                                           f"Original path: '{folder}'. This indicates resolution failed.")
                                self.logger.error(error_msg)
                                raise ValueError(error_msg)
                            
                            resolved_folders.append(resolved)
                        pair['folders'] = resolved_folders
                        self.logger.debug(f"sync_pairs[{i}] resolved: {pair['folders']}")
                else:
                    # Old format (backward compatibility)
                    if 'source' in pair:
                        resolved = self._resolve_path_with_drives(pair['source'], drives)
                        # Validate resolution
                        if not resolved.startswith('/') and not resolved.startswith('~'):
                            first_part = resolved.split('/')[0] if '/' in resolved else resolved
                            if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                               first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                                if first_part not in drives:
                                    error_msg = (f"CRITICAL BUG: Path resolution failed for source '{pair['source']}'. "
                                               f"Resolved to '{resolved}' which still contains unresolved drive placeholder '{first_part}'. "
                                               f"This will create a literal '{first_part}' directory.")
                                    self.logger.error(error_msg)
                                    raise ValueError(error_msg)
                        pair['source'] = resolved
                    if 'target' in pair:
                        resolved = self._resolve_path_with_drives(pair['target'], drives)
                        # Validate resolution
                        if not resolved.startswith('/') and not resolved.startswith('~'):
                            first_part = resolved.split('/')[0] if '/' in resolved else resolved
                            if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                               first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                                if first_part not in drives:
                                    error_msg = (f"CRITICAL BUG: Path resolution failed for target '{pair['target']}'. "
                                               f"Resolved to '{resolved}' which still contains unresolved drive placeholder '{first_part}'. "
                                               f"This will create a literal '{first_part}' directory.")
                                    self.logger.error(error_msg)
                                    raise ValueError(error_msg)
                        pair['target'] = resolved

        # Resolve drive placeholders in one_way_pairs
        if 'one_way_pairs' in self.config:
            self.logger.info(f"Resolving drive placeholders in {len(self.config['one_way_pairs'])} one-way pairs...")
            for i, pair in enumerate(self.config['one_way_pairs']):
                if 'folders' in pair:
                    # New format: resolve each folder in the list
                    if isinstance(pair['folders'], list):
                        resolved_folders = []
                        for folder in pair['folders']:
                            self.logger.debug(f"Resolving path '{folder}' in one_way_pairs[{i}]")
                            resolved = self._resolve_path_with_drives(folder, drives)
                            self.logger.debug(f"  Resolved to: '{resolved}'")
                            
                            # CRITICAL: Validate that resolution actually worked
                            # The resolved path MUST be absolute and MUST NOT contain any drive placeholder strings
                            if not resolved.startswith('/') and not resolved.startswith('~'):
                                first_part = resolved.split('/')[0] if '/' in resolved else resolved
                                if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                                   first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                                    if first_part not in drives:
                                        error_msg = (f"CRITICAL BUG: Path resolution failed for '{folder}' in one_way_pairs[{i}]. "
                                                   f"Resolved to '{resolved}' which still contains unresolved drive placeholder '{first_part}'. "
                                                   f"Available drives: {list(drives.keys())}. "
                                                   f"This will create a literal '{first_part}' directory. "
                                                   f"Please check your drives configuration.")
                                        self.logger.error(error_msg)
                                        raise ValueError(error_msg)
                            
                            # Additional check: resolved path must not contain "MAIN_DRIVE" anywhere
                            if 'MAIN_DRIVE' in resolved:
                                error_msg = (f"CRITICAL BUG: Resolved path '{resolved}' still contains 'MAIN_DRIVE' string. "
                                           f"Original path: '{folder}'. This indicates resolution failed.")
                                self.logger.error(error_msg)
                                raise ValueError(error_msg)
                            
                            resolved_folders.append(resolved)
                        pair['folders'] = resolved_folders
                        self.logger.debug(f"one_way_pairs[{i}] resolved: {pair['folders']}")
                else:
                    # Old format (backward compatibility)
                    if 'source' in pair:
                        resolved = self._resolve_path_with_drives(pair['source'], drives)
                        # Validate resolution
                        if not resolved.startswith('/') and not resolved.startswith('~'):
                            first_part = resolved.split('/')[0] if '/' in resolved else resolved
                            if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                               first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                                if first_part not in drives:
                                    error_msg = (f"CRITICAL BUG: Path resolution failed for source '{pair['source']}'. "
                                               f"Resolved to '{resolved}' which still contains unresolved drive placeholder '{first_part}'. "
                                               f"This will create a literal '{first_part}' directory.")
                                    self.logger.error(error_msg)
                                    raise ValueError(error_msg)
                        pair['source'] = resolved
                    if 'target' in pair:
                        resolved = self._resolve_path_with_drives(pair['target'], drives)
                        # Validate resolution
                        if not resolved.startswith('/') and not resolved.startswith('~'):
                            first_part = resolved.split('/')[0] if '/' in resolved else resolved
                            if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                               first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                                if first_part not in drives:
                                    error_msg = (f"CRITICAL BUG: Path resolution failed for target '{pair['target']}'. "
                                               f"Resolved to '{resolved}' which still contains unresolved drive placeholder '{first_part}'. "
                                               f"This will create a literal '{first_part}' directory.")
                                    self.logger.error(error_msg)
                                    raise ValueError(error_msg)
                        pair['target'] = resolved
    
    def _filter_unavailable_drives(self) -> None:
        """
        After drive placeholder resolution, check which configured drive root paths
        actually exist on this machine.  Any sync_pair, one_way_pair, or source_folder
        that references an unavailable drive is silently dropped with a warning.

        This lets the program run normally when a drive is absent — e.g. GoogleDrive
        was uninstalled, EXTERNAL_DRIVE is unplugged, or ProtonDrive isn't installed.

        Without this, the walk-up heuristic in sync_directories would climb from
        (say) /Users/rod/GoogleDrive/MyFiles/dev all the way up to /Users/rod and
        conclude the drive is "online", then create the missing directories locally
        and copy files into them — silently polluting the home folder.
        """
        drives = self.config.get('drives', {})
        if not drives:
            return

        # ── Which drives are unavailable? ──────────────────────────────────────
        unavailable: Dict[str, str] = {}  # drive_name → resolved path
        for name, raw_path in drives.items():
            if name.startswith('comment') or not isinstance(raw_path, str):
                continue
            resolved = Path(raw_path).expanduser()
            if not resolved.exists():
                unavailable[name] = str(resolved)
                self.logger.warning(
                    f"Drive '{name}' not found at {resolved} — "
                    f"pairs and folders that use it will be skipped"
                )

        if not unavailable:
            return  # All drives present — nothing to filter

        self.logger.warning(
            f"Unavailable drive(s): {list(unavailable.keys())}. "
            f"Sync pairs and source folders that reference them will be skipped this cycle."
        )

        # ── Helper: does a resolved folder path live under an unavailable drive? ─
        def _unavailable_drive_for(folder_path: str) -> Optional[str]:
            for drive_name, drive_root in unavailable.items():
                if folder_path == drive_root or folder_path.startswith(drive_root + '/'):
                    return drive_name
            return None

        # ── Filter sync_pairs ──────────────────────────────────────────────────
        kept_sync = []
        for pair in self.config.get('sync_pairs', []):
            folders = pair.get('folders') or [pair.get('source', ''), pair.get('target', '')]
            bad = next((d for f in folders if f for d in [_unavailable_drive_for(f)] if d), None)
            if bad:
                self.logger.info(f"  Skipping sync pair {folders} — drive '{bad}' not available")
            else:
                kept_sync.append(pair)
        n_dropped = len(self.config.get('sync_pairs', [])) - len(kept_sync)
        self.config['sync_pairs'] = kept_sync

        # ── Filter one_way_pairs ───────────────────────────────────────────────
        kept_one_way = []
        for pair in self.config.get('one_way_pairs', []):
            folders = pair.get('folders') or [pair.get('source', ''), pair.get('target', '')]
            bad = next((d for f in folders if f for d in [_unavailable_drive_for(f)] if d), None)
            if bad:
                self.logger.info(f"  Skipping one-way pair {folders} — drive '{bad}' not available")
            else:
                kept_one_way.append(pair)
        n_dropped += len(self.config.get('one_way_pairs', [])) - len(kept_one_way)
        self.config['one_way_pairs'] = kept_one_way

        # ── Filter source_folders ──────────────────────────────────────────────
        original_sources = self.config.get('source_folders', [])
        kept_sources = []
        for f in original_sources:
            bad = _unavailable_drive_for(f)
            if bad:
                self.logger.info(f"  Skipping source_folder '{f}' — drive '{bad}' not available")
            else:
                kept_sources.append(f)
        self.config['source_folders'] = kept_sources

        if n_dropped:
            self.logger.warning(f"Dropped {n_dropped} pair(s) due to unavailable drives.")

    def _resolve_path(self, path: str) -> str:
        """Resolve a path containing drive placeholders (uses resolved drives)."""
        drives = self.config.get('drives', {})
        # Use the recursive resolver
        return self._resolve_path_with_drives(path, drives)
    
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
        
        # Filter to suppress PyTorch MPS pin_memory warnings
        class PyTorchWarningFilter(logging.Filter):
            def filter(self, record):
                # Suppress PyTorch MPS pin_memory warnings
                if 'pin_memory' in record.getMessage() and 'MPS' in record.getMessage():
                    return False
                return True
        
        pytorch_filter = PyTorchWarningFilter()
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.addFilter(pytorch_filter)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler - truncate log in production mode
            log_file = Path.home() / '.file_organizer.log'
            log_mode = 'a' if self.test_mode else 'w'  # Append in test mode, truncate in production
            file_handler = logging.FileHandler(log_file, mode=log_mode)
            file_handler.addFilter(pytorch_filter)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_progress(self) -> Dict:
        """Load progress from previous run if it exists."""
        if self.test_mode:
            if self.progress_file.exists():
                try:
                    self.progress_file.unlink()
                    self.logger.info("Cleared stale test-mode progress file")
                except Exception as e:
                    self.logger.warning(f"Could not clear test progress file: {e}")
            return {
                'timestamp': time.time(),
                'current_step': 'scan',
                'scan_folders_completed': [],
                'sync_pairs_completed': [],
                'one_way_pairs_completed': [],
                'deduplication_completed': False,
                'backup_completed': False
            }

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
            'one_way_pairs_completed': [],
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
            'one_way_pairs_completed': [],
            'deduplication_completed': False,
            'backup_completed': False
        }

    # ------------------------------------------------------------------
    # State cache helpers (incremental scanning)
    # ------------------------------------------------------------------

    def _get_config_hash(self) -> str:
        """Return an MD5 of the config keys that affect which symlinks are created.

        If any of these change between runs the cached analysis is invalid and
        the whole cache is discarded so every file gets re-processed.
        """
        key_values = {k: self.config.get(k, d) for k, d in [
            ('enable_content_analysis', True),
            ('max_soft_links_per_file', 6),
            ('semantic_confidence_threshold', 0.35),
            ('min_file_size', 1024),
            ('max_file_size', 104857600),
        ]}
        return hashlib.md5(json.dumps(key_values, sort_keys=True).encode()).hexdigest()

    def _load_state_cache(self) -> None:
        """Load the per-file state cache from disk into self._state_cache.

        The cache maps str(absolute_path) → [mtime, size, [rel_link, ...]]
        where rel_link paths are relative to output_base.

        Test mode: always starts fresh (deletes any leftover cache file).
        Production: discards the cache if the version or config_hash changed.
        """
        if self.test_mode:
            if self._state_cache_file.exists():
                try:
                    self._state_cache_file.unlink()
                    self.logger.info("Cleared stale test-mode state cache")
                except Exception as e:
                    self.logger.warning(f"Could not clear test state cache: {e}")
            self._state_cache = {}
            return

        if not self._state_cache_file.exists():
            self._state_cache = {}
            return

        try:
            with open(self._state_cache_file, 'r') as f:
                data = json.load(f)

            if data.get('version') != 1:
                self.logger.info("State cache version mismatch — starting fresh")
                self._state_cache = {}
                return

            if data.get('config_hash') != self._get_config_hash():
                self.logger.info("State cache invalidated (config changed) — starting fresh")
                self._state_cache = {}
                return

            self._state_cache = data.get('files', {})
            self.logger.info(
                f"Loaded state cache: {len(self._state_cache)} entries "
                f"from {self._state_cache_file}"
            )
        except Exception as e:
            self.logger.warning(f"Could not load state cache ({e}) — starting fresh")
            self._state_cache = {}

    def _save_state_cache(self) -> None:
        """Atomically persist self._state_cache to disk.

        Uses a write-to-tmp-then-rename approach so a mid-write crash never
        leaves a corrupt cache file.  Called after each source_folder scan
        completes (same cadence as _save_progress) and after Steps 2/2b.
        """
        try:
            data = {
                'version': 1,
                'config_hash': self._get_config_hash(),
                'files': self._state_cache,
            }
            tmp = self._state_cache_file.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(data, f, separators=(',', ':'))  # compact — no indent
            tmp.rename(self._state_cache_file)
            self.logger.debug(f"State cache saved: {len(self._state_cache)} entries")
        except Exception as e:
            self.logger.warning(f"Could not save state cache: {e}")

    def _remove_file_links(self, file_key: str, old_link_rel_paths: List[str]) -> int:
        """Delete stale symlinks recorded in the cache for a file that has changed.

        Args:
            file_key: str(absolute_path), used as key in file_link_counts.
            old_link_rel_paths: paths relative to output_base stored in the
                cache entry, e.g. ["documents/foo.pdf", "2024/foo.pdf"].

        Returns:
            Number of symlinks successfully removed.
        """
        output_base = Path(self.config['output_base'])
        removed = 0
        for rel_path in old_link_rel_paths:
            link_path = output_base / rel_path
            try:
                if link_path.is_symlink():
                    link_path.unlink()
                    removed += 1
                    self.logger.debug(f"Removed stale link: {link_path}")
            except Exception as e:
                self.logger.warning(f"Could not remove stale link {link_path}: {e}")

        if removed:
            self.file_link_counts[file_key] = max(
                0, self.file_link_counts.get(file_key, 0) - removed
            )
            self.logger.info(
                f"Removed {removed} stale link(s) for changed file: "
                f"{Path(file_key).name}"
            )
        return removed

    def _prune_deleted_files_from_cache(self, scanned_folder: Path) -> None:
        """Remove cache entries for source files deleted since the last scan.

        Called after scan_directory() finishes for scanned_folder. Any file
        whose cache key falls under that folder but was not visited during
        the scan is assumed deleted — its entry is removed, its symlinks in
        output_base are torn down, and any now-empty category directories are
        removed (deepest-first so parents become candidates too).
        """
        folder_prefix = str(scanned_folder) + os.sep
        deleted_keys = [
            key for key in self._state_cache
            if key.startswith(folder_prefix) and key not in self.processed_files
        ]
        if not deleted_keys:
            return

        output_base = Path(self.config['output_base'])
        affected_dirs: set = set()

        for file_key in deleted_keys:
            cached = self._state_cache.pop(file_key)
            old_links = cached[2] if len(cached) > 2 else []
            if old_links:
                self._remove_file_links(file_key, old_links)
                for rel in old_links:
                    affected_dirs.add((output_base / rel).parent)
            self.file_link_counts.pop(file_key, None)

        # Remove empty category directories, deepest first so parent dirs
        # become candidates after their children are removed.
        for d in sorted(affected_dirs, key=lambda p: len(p.parts), reverse=True):
            if d == output_base or not d.is_dir():
                continue
            try:
                if not any(d.iterdir()):
                    d.rmdir()
                    self._cycle_stats['dirs_cleaned'] += 1
                    self.logger.debug(f"Removed empty directory: {d}")
            except OSError:
                pass

        self._cycle_stats['files_pruned'] += len(deleted_keys)
        self.logger.info(
            f"Pruned {len(deleted_keys)} deleted file(s) from cache under {scanned_folder}"
        )

    def _update_state_cache_for_file(
        self, file_key: str, file_path: Path, link_rel_paths: List[str]
    ) -> None:
        """Write (or overwrite) the cache entry for file_key with current stat.

        Called after process_file() finishes analysis (whether or not any
        links were created).  link_rel_paths is [] when no links were made —
        that empty list is what allows Gate 2 in process_file() to skip
        re-analysis on the next run.
        """
        try:
            st = file_path.stat()
            self._state_cache[file_key] = [st.st_mtime, st.st_size, link_rel_paths]
        except Exception as e:
            self.logger.debug(f"Could not stat {file_path} for cache update: {e}")

    def _append_link_to_cache(self, file_key: str, rel_link: str) -> None:
        """Append a link path to an existing cache entry (no-op if entry absent).

        Called from Steps 2 and 2b in run_full_cycle() when keyword/semantic
        category links are created after process_file() has already written
        the initial cache entry.
        """
        entry = self._state_cache.get(file_key)
        if entry is not None and len(entry) > 2:
            if rel_link not in entry[2]:
                entry[2].append(rel_link)

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
        
        # CRITICAL: Always exclude softlink_backup_base (where excluded folders are backed up)
        organised_path = self._get_organised_base()
        try:
            if path.is_relative_to(organised_path) or organised_path in path.parents:
                return True
        except ValueError:
            # Path is not relative, check if organised appears in path
            if 'organised' in path_str and str(organised_path) in path_str:
                return True
        
        # Check exclude_folders (absolute paths) - optional, defaults to empty list
        exclude_folders = self.config.get('exclude_folders', [])
        for exclude in exclude_folders:
            if path_str.startswith(exclude):
                return True
        
        # Check all exclude patterns (exclude_patterns, softlink_folder_patterns, empty_folder_patterns)
        import fnmatch
        exclude_patterns = self.config.get('exclude_patterns', [])
        softlink_patterns = self.config.get('softlink_folder_patterns', [])
        empty_patterns = self.config.get('empty_folder_patterns', [])
        all_patterns = exclude_patterns + softlink_patterns + empty_patterns
        
        path_parts = path.parts
        for pattern in all_patterns:
            if '*' in pattern or '?' in pattern:
                for part in path_parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True
            else:
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
                image = Image.open(file_path)

                # Guard: skip OCR on images too small to contain readable text.
                # PIL.Image.open() is lazy — image.size reads only the header, no pixel decoding.
                # Covers icons, favicons, and thumbnails that would waste EasyOCR/CLIP/Tesseract.
                min_pixels = self.config.get('min_image_pixels_for_ocr', 10_000)  # default 100×100
                if min_pixels > 0:
                    width, height = image.size
                    if width * height < min_pixels:
                        self.logger.debug(
                            f"Skipping OCR on {file_path.name} — image too small "
                            f"({width}×{height}={width*height} px < {min_pixels} px threshold)"
                        )
                        return ""

                all_text = []

                # Method 1: Try EasyOCR first (better with artistic text)
                easyocr_reader = self.content_analyzer.get_easyocr_reader()
                if easyocr_reader:
                    try:
                        import numpy as np
                        import sys
                        from contextlib import redirect_stderr
                        from io import StringIO
                        
                        img_array = np.array(image)
                        # Temporarily suppress stderr to hide PyTorch MPS pin_memory warnings
                        # (These warnings are harmless - pin_memory just isn't supported on MPS)
                        # Note: This may not catch warnings from PyTorch's C++ code or background threads
                        old_stderr = sys.stderr
                        try:
                            sys.stderr = StringIO()
                            ocr_results = easyocr_reader.readtext(img_array)
                        finally:
                            sys.stderr = old_stderr
                        
                        # Extract text from results (format: [(bbox, text, confidence)])
                        easyocr_text = ' '.join([text for (bbox, text, conf) in ocr_results if conf > 0.3])
                        if easyocr_text.strip():
                            all_text.append(easyocr_text)
                            self.logger.info(f"EasyOCR on {file_path.name}: {easyocr_text[:50]}...")
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
                # Guard: skip frame extraction on very small video files (stubs, corrupted
                # containers, or near-empty clips — unlikely to have readable on-screen text).
                min_video_bytes = self.config.get('min_video_size_for_ocr', 100 * 1024)  # default 100 KB
                if min_video_bytes > 0 and file_path.stat().st_size < min_video_bytes:
                    self.logger.debug(
                        f"Skipping frame OCR on {file_path.name} — "
                        f"file too small ({file_path.stat().st_size} B < {min_video_bytes} B threshold)"
                    )
                    return ""
                self.logger.info(f"Extracting text from video: {file_path.name}")
                return self.extract_text_from_video(file_path)
                
        except Exception as e:
            self.logger.debug(f"Could not extract text from {file_path}: {e}")
            
        return ""
    
    def create_soft_link(
        self, source_path: Path, target_folder: str, target_name: str
    ) -> Optional[str]:
        """Create a soft link from source to target folder.

        Returns the link path relative to output_base (e.g. "documents/foo.pdf")
        when a link is created or already exists correctly, None otherwise.
        The relative path is stored in the state cache so stale links can be
        cleaned up when a file changes.
        """
        try:
            file_key = str(source_path)
            current_count = self.file_link_counts[file_key]

            if current_count >= self.max_links_per_file:
                self.logger.debug(
                    f"Skipping link for {source_path.name} - already has "
                    f"{current_count} links (max: {self.max_links_per_file})"
                )
                return None

            target_dir = Path(self.config['output_base']) / target_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / target_name
            rel = str(Path(target_folder) / target_name)  # relative to output_base

            # Check if link already exists and points to correct location
            if target_path.is_symlink():
                try:
                    existing_target = os.readlink(target_path)
                    relative_source = os.path.relpath(source_path, target_dir)
                    if existing_target == relative_source:
                        # Link already exists and is correct
                        return rel
                except Exception:
                    pass
                # Link exists but points elsewhere — decrement the displaced file's
                # count so Gate 1 won't permanently skip it next run.
                try:
                    old_abs = (target_dir / os.readlink(target_path)).resolve()
                    old_key = str(old_abs)
                    if old_key != file_key and old_key in self.file_link_counts:
                        self.file_link_counts[old_key] = max(
                            0, self.file_link_counts[old_key] - 1
                        )
                except Exception:
                    pass
                target_path.unlink()
            elif target_path.exists():
                # Regular file exists — remove it
                target_path.unlink()

            # Create relative symlink
            relative_source = os.path.relpath(source_path, target_dir)
            target_path.symlink_to(relative_source)
            self.file_link_counts[file_key] += 1
            self._cycle_stats['links_created'] += 1
            self.logger.debug(
                f"Created soft link: {target_path} -> {relative_source} "
                f"(file now has {self.file_link_counts[file_key]} links)"
            )
            return rel

        except Exception as e:
            self.logger.error(
                f"Failed to create soft link {source_path} -> "
                f"{target_folder}/{target_name}: {e}"
            )
            return None
    
    def process_file(self, file_path: Path) -> None:
        """Process a single file with dynamic content analysis."""
        if str(file_path) in self.processed_files:
            return

        # Skip macOS metadata files (files starting with ._)
        if file_path.name.startswith('._'):
            return

        if self.should_exclude_path(file_path):
            return

        file_key = str(file_path)

        # Periodic progress indicator — log every 500 files examined.
        self._scan_file_count += 1
        if self._scan_file_count % 500 == 0:
            self.logger.info(f"  ... {self._scan_file_count:,} files examined so far this cycle")

        # Gate 1: file already has symlinks — was fully processed in a previous run.
        if self.file_link_counts.get(file_key, 0) > 0:
            self.processed_files.add(file_key)
            self._cycle_stats['gate1_skipped'] += 1
            return

        # Gates 2 & 3: consult the state cache.
        cached = self._state_cache.get(file_key)
        is_new_file = (cached is None)
        if cached is not None:
            try:
                st = file_path.stat()
                cached_mtime = cached[0]
                cached_size = cached[1]
                old_links = cached[2] if len(cached) > 2 else []

                if abs(st.st_mtime - cached_mtime) < 0.01 and st.st_size == cached_size:
                    # Gate 2: file unchanged since last analysis.
                    if not old_links:
                        # Was analyzed before and produced no links — safe to skip.
                        self.processed_files.add(file_key)
                        self._cycle_stats['gate2_skipped'] += 1
                        return
                    # Cache says links exist but file_link_counts==0 means
                    # ~/organized was cleared externally.  Fall through to
                    # re-analyze; don't try to remove the already-gone links.
                else:
                    # Gate 3: file has changed — tear down stale links then re-analyze.
                    if old_links:
                        self._remove_file_links(file_key, old_links)
                    del self._state_cache[file_key]
                    self._cycle_stats['gate3_updated'] += 1
            except OSError:
                return  # file vanished between iterdir and stat

        try:
            # Extract content for analysis
            content = ""
            if self.config.get('enable_content_analysis', True):
                content = self.extract_text_content(file_path)

            # Analyze file with dynamic content analyzer
            keywords = self.content_analyzer.analyze_file(file_path, content)

            # Classify file by type and year
            classifications = set()
            classifications.update(self.classify_by_type(file_path))
            classifications.update(self.classify_by_year(file_path))

            # Create soft links and collect their relative paths for the cache.
            new_link_rel_paths: List[str] = []
            for classification in classifications:
                rel = self.create_soft_link(file_path, classification, file_path.name)
                if rel is not None:
                    new_link_rel_paths.append(rel)

            self.processed_files.add(file_key)
            self._newly_processed_files.add(file_key)
            if is_new_file:
                self._cycle_stats['new_files'] += 1

            # Persist analysis result so next run can skip this file if unchanged.
            self._update_state_cache_for_file(file_key, file_path, new_link_rel_paths)

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
    
    def scan_directory(self, directory: Path, max_depth: int = 10) -> None:
        """Scan directory and process files."""
        if not directory.exists() or self.should_exclude_path(directory):
            return
        
        # Check if we should stop
        if hasattr(self, 'running') and not self.running:
            return
        
        self.logger.debug(f"Scanning directory: {directory}")

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
    
    def _sync_pair_list(self, pair_list: List[Dict], progress_key: str, sync_mode: str, pair_label: str, direction_arrow: str) -> None:
        """Sync a list of folder pairs with the specified mode."""
        completed_pairs = self.progress.get(progress_key, [])
        
        for i, sync_pair in enumerate(pair_list):
            # Skip already completed pairs
            if i in completed_pairs:
                continue
            
            # Support both new format (folders) and old format (source/target) for backward compatibility
            if 'folders' in sync_pair:
                # New format: folders is a list of two paths
                folders = sync_pair['folders']
                if not isinstance(folders, list) or len(folders) != 2:
                    self.logger.warning(f"{pair_label}[{i}]: 'folders' must be a list of exactly 2 paths")
                    continue
                folder_a_path = folders[0]
                folder_b_path = folders[1]
            elif 'source' in sync_pair and 'target' in sync_pair:
                # Old format: source/target (backward compatibility)
                folder_a_path = sync_pair['source']
                folder_b_path = sync_pair['target']
            else:
                # Comment-only entry
                self.logger.debug(f"Skipping {pair_label}[{i}] - comment-only entry")
                continue
            
            # Paths should already be resolved from drive placeholders, just expand ~ and resolve
            # CRITICAL: Validate that paths don't contain unresolved drive placeholders before creating Path objects
            # This prevents creating literal "MAIN_DRIVE" directories
            if isinstance(folder_a_path, str):
                # Check if path is relative and contains what looks like a drive placeholder
                if not folder_a_path.startswith('/') and not folder_a_path.startswith('~'):
                    first_part = folder_a_path.split('/')[0]
                    # Check if it looks like a drive placeholder
                    if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                       first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                        drives = self.config.get('drives', {})
                        if first_part not in drives:
                            error_msg = (f"CRITICAL BUG: Unresolved drive placeholder '{first_part}' detected in sync path '{folder_a_path}'. "
                                       f"This would create a literal '{first_part}' directory at the current working directory. "
                                       f"Available drives: {list(drives.keys())}. "
                                       f"Please ensure '{first_part}' is properly defined in the 'drives' section of your config file.")
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
            
            if isinstance(folder_b_path, str):
                # Check if path is relative and contains what looks like a drive placeholder
                if not folder_b_path.startswith('/') and not folder_b_path.startswith('~'):
                    first_part = folder_b_path.split('/')[0]
                    # Check if it looks like a drive placeholder
                    if (first_part.isupper() and ('_' in first_part or first_part.endswith('_DRIVE'))) or \
                       first_part in ['MAIN_DRIVE', 'EXTERNAL_DRIVE', 'GOOGLE_DRIVE', 'PROTON_DRIVE', 'BACKUP_DRIVE']:
                        drives = self.config.get('drives', {})
                        if first_part not in drives:
                            error_msg = (f"CRITICAL BUG: Unresolved drive placeholder '{first_part}' detected in sync path '{folder_b_path}'. "
                                       f"This would create a literal '{first_part}' directory at the current working directory. "
                                       f"Available drives: {list(drives.keys())}. "
                                       f"Please ensure '{first_part}' is properly defined in the 'drives' section of your config file.")
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
            
            # Log the paths before creating Path objects for debugging
            self.logger.debug(f"Creating Path objects - folder_a_path: '{folder_a_path}', folder_b_path: '{folder_b_path}'")
            
            folder_a = Path(folder_a_path).expanduser().resolve()
            folder_b = Path(folder_b_path).expanduser().resolve()
            
            # Log the resolved paths
            self.logger.debug(f"Resolved paths - folder_a: '{folder_a}', folder_b: '{folder_b}'")
            
            # CRITICAL: Check if resolved paths contain MAIN_DRIVE as a directory component
            if 'MAIN_DRIVE' in folder_a.parts:
                error_msg = (f"CRITICAL BUG: Resolved folder_a path contains literal 'MAIN_DRIVE' directory: {folder_a}. "
                           f"Original path: '{folder_a_path}'. Path parts: {folder_a.parts}. "
                           f"This will create a literal MAIN_DRIVE folder. Current working directory: {Path.cwd()}")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if 'MAIN_DRIVE' in folder_b.parts:
                error_msg = (f"CRITICAL BUG: Resolved folder_b path contains literal 'MAIN_DRIVE' directory: {folder_b}. "
                           f"Original path: '{folder_b_path}'. Path parts: {folder_b.parts}. "
                           f"This will create a literal MAIN_DRIVE folder. Current working directory: {Path.cwd()}")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Additional safety check: ensure resolved paths are absolute
            if not folder_a.is_absolute():
                error_msg = f"CRITICAL: Resolved path '{folder_a}' is not absolute. Original: '{folder_a_path}'. This will cause incorrect behavior."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if not folder_b.is_absolute():
                error_msg = f"CRITICAL: Resolved path '{folder_b}' is not absolute. Original: '{folder_b_path}'. This will cause incorrect behavior."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check if folders are the same
            if folder_a == folder_b:
                self.logger.warning(f"{pair_label}[{i}]: Folders are identical, skipping")
                continue
            
            # Check if either folder's drive/path is available
            def check_path_available(path: Path) -> bool:
                parent = path
                while parent.parent != parent and not parent.exists():
                    parent = parent.parent
                return parent.exists()
            
            if sync_mode == 'bidirectional':
                if not check_path_available(folder_a) and not check_path_available(folder_b):
                    self.logger.warning(f"Both folders unavailable: {folder_a} and {folder_b}")
                    self.logger.warning(f"Skipping {pair_label} {i}")
                    # Mark as completed so we don't get stuck
                    completed_pairs.append(i)
                    self.progress[progress_key] = completed_pairs
                    self._save_progress()
                    continue
            else:
                if not check_path_available(folder_a):
                    self.logger.warning(f"Source folder unavailable: {folder_a}")
                    self.logger.info(f"Will retry {pair_label} {i+1} on next run when source becomes available")
                    continue
            
            # CRITICAL: Validate that folder_a and folder_b don't contain MAIN_DRIVE before proceeding
            if 'MAIN_DRIVE' in folder_a.parts:
                error_msg = (f"CRITICAL BUG: folder_a contains literal 'MAIN_DRIVE' directory: {folder_a}. "
                           f"Original path: '{folder_a_path}'. Path parts: {folder_a.parts}. "
                           f"This sync pair is misconfigured and will create incorrect directory structures.")
                self.logger.error(error_msg)
                self.logger.error(f"Skipping {pair_label} {i+1} to prevent data corruption")
                continue
            if 'MAIN_DRIVE' in folder_b.parts:
                error_msg = (f"CRITICAL BUG: folder_b contains literal 'MAIN_DRIVE' directory: {folder_b}. "
                           f"Original path: '{folder_b_path}'. Path parts: {folder_b.parts}. "
                           f"This sync pair is misconfigured and will create incorrect directory structures.")
                self.logger.error(error_msg)
                self.logger.error(f"Skipping {pair_label} {i+1} to prevent data corruption")
                continue
            
            # At least one folder exists, proceed with sync
            if (sync_mode == 'bidirectional' and (folder_a.exists() or folder_b.exists())) or \
               (sync_mode != 'bidirectional' and folder_a.exists()):
                # Use the existing folder as base for chunking if it exists
                base_folder = folder_a if sync_mode != 'bidirectional' else (folder_a if folder_a.exists() else folder_b)
                
                # CRITICAL: Also validate base_folder doesn't contain MAIN_DRIVE
                if 'MAIN_DRIVE' in base_folder.parts:
                    error_msg = (f"CRITICAL BUG: base_folder contains literal 'MAIN_DRIVE' directory: {base_folder}. "
                               f"This indicates the sync source is already in the wrong location. "
                               f"Path parts: {base_folder.parts}. Skipping this sync pair.")
                    self.logger.error(error_msg)
                    self.logger.error(f"You may need to manually clean up the incorrectly created MAIN_DRIVE directory at: {Path.cwd() / 'MAIN_DRIVE'}")
                    continue
                
                # Check if we should chunk this sync (large folders)
                chunk_threshold = self.config.get('sync_chunk_subfolders', 30)
                
                if base_folder.is_dir():
                    # Count immediate subfolders
                    try:
                        subfolders = [d for d in base_folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
                        num_subfolders = len(subfolders)
                        
                        if num_subfolders >= chunk_threshold:
                            # Sync subfolders with limited concurrency
                            max_workers = max(1, int(self.config.get('sync_chunk_concurrency', 2)))
                            
                            def sync_one(subfolder_path: Path):
                                try:
                                    if hasattr(self, 'running') and not self.running:
                                        return False
                                    
                                    # CRITICAL: Check if subfolder_path contains MAIN_DRIVE
                                    if 'MAIN_DRIVE' in subfolder_path.parts:
                                        error_msg = (f"CRITICAL BUG: Subfolder path contains MAIN_DRIVE: {subfolder_path}. "
                                                   f"Path parts: {subfolder_path.parts}. Skipping this subfolder.")
                                        self.logger.error(error_msg)
                                        return False
                                    
                                    # Determine which folder this subfolder belongs to
                                    try:
                                        rel_path = subfolder_path.relative_to(folder_a)
                                        other_subfolder = folder_b / rel_path
                                    except ValueError:
                                        if sync_mode == 'bidirectional':
                                            # Not relative to folder_a, try folder_b
                                            try:
                                                rel_path = subfolder_path.relative_to(folder_b)
                                                other_subfolder = folder_a / rel_path
                                            except ValueError:
                                                self.logger.error(f"Subfolder {subfolder_path} is not relative to either sync folder")
                                                return False
                                        else:
                                            self.logger.error(f"Subfolder {subfolder_path} is not relative to the source folder")
                                            return False
                                    
                                    # CRITICAL: Check if other_subfolder contains MAIN_DRIVE before syncing
                                    if 'MAIN_DRIVE' in other_subfolder.parts:
                                        error_msg = (f"CRITICAL BUG: Calculated other_subfolder contains MAIN_DRIVE: {other_subfolder}. "
                                                   f"Path parts: {other_subfolder.parts}. "
                                                   f"Original subfolder: {subfolder_path}, folder_a: {folder_a}, folder_b: {folder_b}")
                                        self.logger.error(error_msg)
                                        return False
                                    
                                    return self.folder_sync.sync_directories(subfolder_path, other_subfolder, sync_mode=sync_mode)
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
                                        if ok:
                                            short_path = self.folder_sync._shorten_path(sf, [folder_a, folder_b])
                                            self.logger.info(f"Synced subfolder: {short_path}")
                                        else:
                                            short_path = self.folder_sync._shorten_path(sf, [folder_a, folder_b])
                                            self.logger.warning(f"Subfolder sync failed: {short_path}")
                                    except Exception as e:
                                        short_path = self.folder_sync._shorten_path(sf, [folder_a, folder_b])
                                        self.logger.error(f"Subfolder sync error: {short_path} - {e}")
                            
                            # Sync root level files (if any)
                            if base_folder.exists():
                                root_files = [f for f in base_folder.iterdir() if f.is_file()]
                                if root_files:
                                    self.folder_sync.sync_directories(folder_a, folder_b, sync_mode=sync_mode)
                            
                            # Mark chunked sync as completed if all chunks succeeded
                            completed_pairs.append(i)
                            self.progress[progress_key] = completed_pairs
                            self.progress['current_step'] = 'sync'
                            self._save_progress()
                        else:
                            # Normal sync for smaller folders
                            short_a = self.folder_sync._shorten_path(folder_a, [folder_a, folder_b])
                            short_b = self.folder_sync._shorten_path(folder_b, [folder_a, folder_b])
                            self.logger.info(f"Syncing folders: {short_a} {direction_arrow} {short_b}")
                            sync_success = self.folder_sync.sync_directories(folder_a, folder_b, sync_mode=sync_mode)
                            if sync_success:
                                completed_pairs.append(i)
                                self.progress[progress_key] = completed_pairs
                                self.progress['current_step'] = 'sync'
                                self._save_progress()
                            else:
                                self.logger.warning(f"Sync failed for {pair_label} {i+1}, will retry on next run")
                    except Exception as e:
                        self.logger.error(f"Error analyzing folder {base_folder}: {e}")
                        # Fall back to normal sync
                        short_a = self.folder_sync._shorten_path(folder_a, [folder_a, folder_b])
                        short_b = self.folder_sync._shorten_path(folder_b, [folder_a, folder_b])
                        self.logger.info(f"Syncing folders: {short_a} {direction_arrow} {short_b}")
                        sync_success = self.folder_sync.sync_directories(folder_a, folder_b, sync_mode=sync_mode)
                        if sync_success:
                            completed_pairs.append(i)
                            self.progress[progress_key] = completed_pairs
                            self.progress['current_step'] = 'sync'
                            self._save_progress()
                        else:
                            self.logger.warning(f"Sync failed for {pair_label} {i+1}, will retry on next run")
                else:
                    # Not a directory, sync normally
                    short_a = self.folder_sync._shorten_path(folder_a, [folder_a, folder_b])
                    short_b = self.folder_sync._shorten_path(folder_b, [folder_a, folder_b])
                    self.logger.info(f"Syncing folders: {short_a} {direction_arrow} {short_b}")
                    sync_success = self.folder_sync.sync_directories(folder_a, folder_b, sync_mode=sync_mode)
                    if sync_success:
                        completed_pairs.append(i)
                        self.progress[progress_key] = completed_pairs
                        self.progress['current_step'] = 'sync'
                        self._save_progress()
                    else:
                        self.logger.warning(f"Sync failed for {pair_label} {i+1}, will retry on next run")
            else:
                if sync_mode == 'bidirectional':
                    self.logger.warning(f"Neither folder exists: {folder_a} or {folder_b}")
                    # Don't mark as completed if folders don't exist - we want to retry
                    self.logger.info(f"Will retry {pair_label} {i+1} on next run when folders become available")
                else:
                    self.logger.warning(f"Source folder does not exist: {folder_a}")
                    self.logger.info(f"Will retry {pair_label} {i+1} on next run when source becomes available")
    
    def sync_configured_folders(self) -> None:
        """Synchronize configured folder pairs."""
        if not self.config['enable_folder_sync']:
            return
        
        self.logger.info("Starting folder synchronization")
        
        self._sync_pair_list(
            self.config.get('sync_pairs', []),
            progress_key='sync_pairs_completed',
            sync_mode='bidirectional',
            pair_label='sync_pairs',
            direction_arrow='↔'
        )
        self._sync_pair_list(
            self.config.get('one_way_pairs', []),
            progress_key='one_way_pairs_completed',
            sync_mode='source',
            pair_label='one_way_pairs',
            direction_arrow='→'
        )
    
    def remove_duplicates_across_system(self) -> None:
        """Find and remove duplicates across all source folders."""
        if not self.config['enable_duplicate_detection']:
            return
        
        # Skip if already completed
        if self.progress.get('deduplication_completed', False):
            self.logger.info("✓ Skipping deduplication - already completed in this run")
            return
        
        self.logger.info("Scanning for duplicate files")
        
        # Get source folders (use empty list if not configured)
        source_folders = self.config.get('source_folders', [])
        if not source_folders:
            self.logger.warning("No source_folders configured - skipping duplicate detection")
            return
        
        directories = [Path(folder) for folder in source_folders]
        duplicates = self.folder_sync.find_duplicates_in_directories(directories)
        
        if duplicates:
            self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
            dry_run = self.config.get('dedupe_dry_run', True)
            if dry_run:
                self.logger.info("Deduplication running in DRY RUN mode — no files will be deleted. "
                                 "Set 'dedupe_dry_run: false' in config to enable real deletion.")
            removed = self.folder_sync.remove_duplicates(duplicates, keep_newest=True, dry_run=dry_run)
            action = "Would remove" if dry_run else "Removed"
            self.logger.info(f"{action} {removed} duplicate files")
        
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
        # Ensure progress is initialized (safety check)
        if not hasattr(self, 'progress') or self.progress is None:
            self.progress = self._load_progress()
        
        # Reset per-cycle stats and progress counter
        self._cycle_stats = {k: 0 for k in self._cycle_stats}
        self._scan_file_count = 0

        self.logger.info("=" * 80)
        self.logger.info("Starting full organization cycle")
        self.logger.info("=" * 80)
        
        # Step 1: Scan and organize files FIRST (gives immediate value before time-consuming operations)
        self.logger.info("Step 1: Scanning and organizing files...")
        
        # Get source folders (use empty list if not configured)
        scan_folders = self.config.get('source_folders', [])
        completed_scan_folders = self.progress.get('scan_folders_completed', [])
        
        if not scan_folders:
            self.logger.info("No source_folders configured - skipping file scanning")
            self.progress['scan_folders_completed'] = []
            self._save_progress()
            # Continue to next step (sync, dedupe, etc.)
        else:
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
                    self._prune_deleted_files_from_cache(folder_path)

                    # Mark this folder as completed
                    completed_scan_folders.append(i)
                    self.progress['scan_folders_completed'] = completed_scan_folders
                    self.progress['current_step'] = 'scan'
                    self._save_progress()
                    self._save_state_cache()
        
        # Step 2: Discover content categories based on learned keywords
        if hasattr(self, 'running') and not self.running:
            return
        if self.config.get('ml_content_analysis', {}).get('enabled', True):
            self.logger.info("Step 2: Discovering content categories from analyzed files...")
            # Reset category tracking before discovering categories (individual keywords already tracked)
            # Note: discovered_categories() already respects max_categories internally
            discovered_categories = self.content_analyzer.discover_categories()
            
            # Create soft links for discovered categories
            # (discovered_categories() already respects max_categories internally)
            for category_name, file_paths in discovered_categories.items():
                if hasattr(self, 'running') and not self.running:
                    return
                for file_path_str in file_paths:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        rel = self.create_soft_link(file_path, category_name, file_path.name)
                        if rel:
                            self._append_link_to_cache(file_path_str, rel)
            
            # Save discovered categories for review
            categories_file = Path.home() / '.file_organizer_discovered_categories.json'
            self.content_analyzer.save_discovered_categories(categories_file)
            self.logger.info(f"Discovered {len(discovered_categories)} content categories")
        
        # Step 2b: Semantic categorization — group files into broad categories
        if hasattr(self, 'running') and not self.running:
            return
        if self.config.get('enable_semantic_categories', True):
            new_file_keywords = {
                fp: kws for fp, kws in self.content_analyzer.file_keywords.items()
                if fp in self._newly_processed_files
            }
            if new_file_keywords:
                self.logger.info(f"Step 2b: Classifying {len(new_file_keywords)} new files into semantic categories...")
                semantic_groups = self.semantic_categorizer.group_files(new_file_keywords)
                for category_name, file_paths in semantic_groups.items():
                    if hasattr(self, 'running') and not self.running:
                        return
                    for fp_str in file_paths:
                        fp = Path(fp_str)
                        if fp.exists():
                            rel = self.create_soft_link(fp, category_name, fp.name)
                            if rel:
                                self._append_link_to_cache(fp_str, rel)
                self.logger.info(f"Created semantic links across {len(semantic_groups)} broad categories")
            else:
                self.logger.info("Step 2b: No new files to classify semantically")
        
        self._save_state_cache()
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

        # Print end-of-cycle summary
        s = self._cycle_stats
        total_scanned = (
            s['gate1_skipped'] + s['gate2_skipped'] +
            s['gate3_updated'] + s['new_files']
        )
        total_with_links = len([k for k, v in self.file_link_counts.items() if v > 0])
        total_links = sum(self.file_link_counts.values())
        output_base = self.config.get('output_base', '~/organized')
        self.logger.info("=" * 80)
        self.logger.info("Full organization cycle complete")
        self.logger.info("-" * 40)
        self.logger.info(f"  Files scanned:            {total_scanned:>8,}")
        self.logger.info(f"    Already organized:      {s['gate1_skipped']:>8,}")
        self.logger.info(f"    Unchanged, skipped:     {s['gate2_skipped']:>8,}")
        self.logger.info(f"    Updated (changed):      {s['gate3_updated']:>8,}")
        self.logger.info(f"    New:                    {s['new_files']:>8,}")
        self.logger.info(f"  Links created this run:   {s['links_created']:>8,}")
        self.logger.info(f"  Files pruned (deleted):   {s['files_pruned']:>8,}")
        if s['dirs_cleaned']:
            self.logger.info(f"  Empty dirs removed:       {s['dirs_cleaned']:>8,}")
        self.logger.info(f"  Unique files with links:  {total_with_links:>8,}")
        self.logger.info(f"  Total symlinks:           {total_links:>8,}")
        self.logger.info(f"  Output: {output_base}")
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


def validate_config_file(config_file: str) -> bool:
    """
    Validate the YAML config file before starting the organizer.
    Returns True if valid, prints error and exits with code 1 if invalid.
    This is called before daemonizing so errors are shown immediately.
    """
    if not os.path.exists(config_file):
        print("\n" + "=" * 70)
        print("ERROR: Configuration file not found!")
        print("=" * 70)
        print(f"\nThe file '{config_file}' does not exist.")
        print("\nA valid configuration file is required, even in test mode.")
        
        template_file = "config_template.yaml"
        if os.path.exists(template_file):
            print("\nTo get started:")
            print(f"  1. Copy the template:  cp {template_file} {config_file}")
            print(f"  2. Edit your config:   nano {config_file}")
            print("  3. Update the 'drives' section with your actual paths")
            print(f"  4. Run the program again")
        else:
            print(f"\nPlease create {config_file} with your drive configurations.")
        
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)
    
    # Check for TAB characters in the file (YAML doesn't allow tabs)
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                if '\t' in line:
                    print("\n" + "=" * 70)
                    print("INVALID YAML CONFIGURATION")
                    print("=" * 70)
                    print(f"\nThe file '{config_file}' contains TAB characters on line {line_num}.")
                    print("\nYAML does not allow TAB characters for indentation. You must use SPACES instead.")
                    print("\nThis is a common mistake when editing YAML files.")
                    print("\nTo fix this:")
                    print("  1. Open the file in your editor")
                    print(f"  2. Find line {line_num} and replace any TAB characters with spaces")
                    print("  3. Most editors have a 'Show Whitespace' or 'Show Invisibles' option")
                    print("  4. You can also use: sed -i '' 's/\\t/  /g' " + config_file)
                    print("\n" + "=" * 70 + "\n")
                    sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Could not read configuration file")
        print("=" * 70)
        print(f"\nFailed to read '{config_file}':")
        print(f"  {e}")
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)
    
    # Try to load and parse the YAML file
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print("\n" + "=" * 70)
            print("INVALID YAML CONFIGURATION")
            print("=" * 70)
            print(f"\nThe file '{config_file}' appears to be empty or contains no valid YAML.")
            print("\nPlease ensure the file contains valid YAML configuration.")
            print("\n" + "=" * 70 + "\n")
            sys.exit(1)
        
        # Basic validation - just check it's a dict
        if not isinstance(config, dict):
            print("\n" + "=" * 70)
            print("INVALID YAML CONFIGURATION")
            print("=" * 70)
            print(f"\nThe file '{config_file}' does not contain a valid configuration dictionary.")
            print("\n" + "=" * 70 + "\n")
            sys.exit(1)
        
        return True
        
    except yaml.YAMLError as e:
        # YAML syntax error - provide user-friendly error messages
        print("\n" + "=" * 70)
        print("INVALID YAML CONFIGURATION")
        print("=" * 70)
        
        error_msg = str(e)
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            line_num = mark.line + 1
            col_num = mark.column + 1
            print(f"\nThe file '{config_file}' contains invalid YAML at line {line_num}, column {col_num}:")
            print(f"  {error_msg}")
            
            # Check for common indentation issues
            if 'indentation' in error_msg.lower() or 'expected' in error_msg.lower():
                print("\nThis looks like an indentation error. YAML is very sensitive to indentation.")
                print("\nCommon causes:")
                print("  - Mixing spaces and tabs (YAML requires spaces only)")
                print("  - Incorrect number of spaces for indentation")
                print("  - Inconsistent indentation levels")
                print("\nTo fix this:")
                print(f"  1. Check line {line_num} in your config file")
                print("  2. Ensure you're using SPACES (not tabs) for indentation")
                print("  3. YAML typically uses 2 spaces per indentation level")
                print("  4. Make sure all items at the same level use the same indentation")
            else:
                print("\nThis is a YAML syntax error.")
                print("\nTo fix this:")
                print("  1. Check the line and column mentioned above")
                print("  2. Ensure proper YAML syntax (colons, dashes, quotes, etc.)")
                print("  3. Use a YAML validator: https://www.yamllint.com/")
        else:
            print(f"\nThe file '{config_file}' contains invalid YAML:")
            print(f"  {error_msg}")
            print("\nThis is a common mistake when editing YAML configuration files.")
            print("\nTo fix this:")
            print("  1. Check for syntax errors (missing colons, incorrect indentation, etc.)")
            print("  2. Use a YAML validator: https://www.yamllint.com/")
            print("  3. Or start fresh: cp config_template.yaml " + config_file)
        
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Could not validate configuration file")
        print("=" * 70)
        print(f"\nFailed to validate '{config_file}':")
        print(f"  {e}")
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)


def main():
    """Main entry point."""
    # Bootstrap: configure logging immediately so daemon mode writes to log file from the start
    log_file = Path.home() / '.file_organizer.log'
    bootstrap_logger = logging.getLogger('file_organizer')
    if not bootstrap_logger.handlers:
        bootstrap_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        bootstrap_logger.addHandler(handler)
    bootstrap_logger.info("Starting file organizer...")

    parser = argparse.ArgumentParser(
        description='File Organizer using AI/ML',
        epilog='Run without arguments for TEST MODE (foo/bar/baz folders). Use --REAL for PRODUCTION MODE.'
    )
    parser.add_argument('-R', '--REAL', action='store_true',
                       help='Run in PRODUCTION MODE on your entire file system (default: TEST MODE)')
    parser.add_argument('--config', default=None,
                       help='Configuration file path (default: config.yaml)')
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
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate configuration file and exit (does not run organizer)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove broken and now-excluded symlinks from ~/organized, then exit')
    # Internal flag used after relaunching as a background process to suppress console output
    parser.add_argument('--internal-daemon', action='store_true', help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # If --validate-config, validate and exit early (skip other checks)
    if args.validate_config:
        config_file = args.config or 'config.yaml'
        validate_config_file(config_file)
        print("\n✓ Configuration file is valid!")
        sys.exit(0)
    
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
        config_file = args.config or 'config.yaml'
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
        
        # In test mode, still require config.yaml (even though test mode doesn't use it)
        config_file = args.config or 'config.yaml'
    
    # Validate config file BEFORE daemonizing (so errors show immediately)
    # This function will exit with code 1 if validation fails
    validate_config_file(config_file)
    
    # In production mode, if not a one-shot scan and not explicitly disabled, daemonize
    if production_mode and not args.scan_once and not args.no_daemon and not args.internal_daemon:
        try:
            # Redirect stderr to log file so errors are captured
            log_file = Path.home() / '.file_organizer.log'
            log_handle = open(log_file, 'a')
            background_args = [sys.executable, __file__] + [a for a in sys.argv[1:] if a != '--force'] + ['--internal-daemon']
            subprocess.Popen(background_args, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            log_handle.close()
            print("Started File Organizer in background (daemon mode).")
            print("View logs:   ./manage_organizer.sh log")
            print("Stop daemon: ./manage_organizer.sh stop")
            return
        except Exception as e:
            print(f"Failed to start in background: {e}")
            import traceback
            traceback.print_exc()
            # Fall through and run in foreground

    # Create organizer
    try:
        organizer = EnhancedFileOrganizer(config_file, test_mode=not production_mode)
    except Exception as e:
        print(f"ERROR: Failed to initialize File Organizer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Handle cleanup mode
    if args.cleanup:
        organizer.cleanup_organized()
        return

    # Execute based on options
    try:
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
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

