#!/usr/bin/env python3
"""
Cleanup script for file-organiser.

This script:
1. Removes ._ soft links from ~/organized
2. Restores incorrectly softlinked folders (folders that were softlinked but shouldn't have been)

WARNING: This script will DELETE soft links and restore folders. Run with care.
"""

import os
import yaml
import shutil
from pathlib import Path

def load_config():
    """Load config.yaml to get softlink_folder_patterns and softlink_backup_base."""
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        print(f"Warning: {config_path} not found, using defaults")
        return {
            'softlink_folder_patterns': ['.git', '.hg', '.svn', '.cvs', '__pycache__', '.pytest_cache', 
                                         '.mypy_cache', '.tox', '.venv', 'venv', 'env'],
            'softlink_backup_base': '~/organised'
        }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            config = {}
        return {
            'softlink_folder_patterns': config.get('softlink_folder_patterns', []),
            'softlink_backup_base': config.get('softlink_backup_base', '~/organised')
        }
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        return {
            'softlink_folder_patterns': ['.git', '.hg', '.svn', '.cvs', '__pycache__', '.pytest_cache', 
                                         '.mypy_cache', '.tox', '.venv', 'venv', 'env'],
            'softlink_backup_base': '~/organised'
        }

def folder_matches_pattern(folder_name: str, patterns: list) -> bool:
    """Check if folder name exactly matches any pattern (no substring matching)."""
    for pattern in patterns:
        if folder_name == pattern:
            return True
        # Handle wildcard patterns
        if '*' in pattern or '?' in pattern:
            import fnmatch
            if fnmatch.fnmatch(folder_name, pattern):
                return True
    return False

def cleanup_dot_underscore_links(organized_path: Path) -> int:
    """Find and delete all soft links starting with '._' in ~/organized. Returns count."""
    deleted_count = 0
    
    if not organized_path.exists():
        print(f"Warning: {organized_path} does not exist, skipping ._ cleanup")
        return 0
    
    # Walk through all files recursively
    for root, dirs, files in os.walk(organized_path):
        directory = Path(root)
        
        for item in directory.iterdir():
            # Check if it's a file (including symlinks) starting with '._'
            if item.name.startswith('._'):
                if item.is_file() or item.is_symlink():
                    try:
                        print(f"Deleting ._ link: {item}")
                        item.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {item}: {e}")
    
    return deleted_count

def restore_incorrectly_linked_folders(organised_base: Path, softlink_patterns: list, search_paths: list = None) -> int:
    """
    Find and restore incorrectly softlinked folders.
    
    A folder is incorrectly linked if:
    - It's a symlink pointing to softlink_backup_base
    - Its name doesn't match any pattern in softlink_folder_patterns
    
    Returns count of restored folders.
    """
    restored_count = 0
    
    if search_paths is None:
        # Default search paths - common development directories
        search_paths = [
            Path.home() / 'dev',
            Path.home() / 'Documents',
            Path.home() / 'Projects',
        ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        print(f"\nScanning {search_path} for incorrectly linked folders...")
        
        # Walk through directories recursively
        for root, dirs, files in os.walk(search_path):
            # Don't modify dirs list while iterating
            for item_name in list(dirs):
                item_path = Path(root) / item_name
                
                # Check if it's a symlink pointing to organised
                if item_path.is_symlink():
                    try:
                        link_target = item_path.readlink()
                        link_target_str = str(link_target)
                        
                        # Check if it points to organised
                        if str(organised_base) in link_target_str or 'organised' in link_target_str:
                            # Check if folder name matches a valid pattern
                            if not folder_matches_pattern(item_name, softlink_patterns):
                                # This is incorrectly linked - restore it
                                print(f"\nFound incorrectly linked folder: {item_path}")
                                print(f"  Points to: {link_target}")
                                print(f"  Folder name '{item_name}' doesn't match any pattern in softlink_folder_patterns")
                                
                                # Restore from backup
                                if link_target.exists() and link_target.is_dir():
                                    try:
                                        # Remove the symlink
                                        item_path.unlink()
                                        # Copy the folder back from backup
                                        shutil.copytree(link_target, item_path)
                                        print(f"  ✓ Restored {item_path} from backup")
                                        restored_count += 1
                                    except Exception as e:
                                        print(f"  ✗ Error restoring {item_path}: {e}")
                                else:
                                    print(f"  ✗ Backup location {link_target} doesn't exist, cannot restore")
                                    print(f"    You may need to manually remove the symlink: {item_path}")
                    except Exception as e:
                        # Skip if we can't read the symlink
                        continue
    
    return restored_count

def main():
    organized_path = Path.home() / 'organized'
    
    # Load config
    config = load_config()
    softlink_patterns = config['softlink_folder_patterns']
    backup_base_str = config['softlink_backup_base']
    
    # Expand ~ to home directory
    if backup_base_str.startswith('~'):
        backup_base_str = str(Path.home()) + backup_base_str[1:]
    organised_base = Path(backup_base_str)
    
    print(f"Valid softlink patterns: {softlink_patterns}")
    print(f"Softlink backup base: {organised_base}")
    
    print("\n" + "="*70)
    print("File Organizer Cleanup Script")
    print("="*70)
    print("\nThis script will:")
    print("1. Delete ._ soft links from ~/organized")
    print("2. Restore incorrectly softlinked folders (those that don't match softlink_folder_patterns)")
    print("\nWARNING: This will DELETE soft links and RESTORE folders from backups.")
    print("Continue? (yes/no): ", end='')
    
    response = input().strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Step 1: Clean up ._ links in ~/organized
    print("\n" + "="*70)
    print("Step 1: Cleaning up ._ links in ~/organized")
    print("="*70)
    dot_underscore_count = cleanup_dot_underscore_links(organized_path)
    print(f"\nDeleted {dot_underscore_count} ._ soft link(s).")
    
    # Step 2: Restore incorrectly linked folders
    print("\n" + "="*70)
    print("Step 2: Restoring incorrectly softlinked folders")
    print("="*70)
    print("\nThis will scan common directories for incorrectly linked folders.")
    print("You can specify custom search paths if needed.")
    print("\nScan default paths? (yes/no): ", end='')
    
    response = input().strip().lower()
    if response in ['yes', 'y']:
        restored_count = restore_incorrectly_linked_folders(organised_base, softlink_patterns)
        print(f"\n✓ Restored {restored_count} incorrectly linked folder(s).")
    else:
        print("Skipped folder restoration.")
    
    print("\n" + "="*70)
    print("Cleanup complete!")
    print("="*70)

if __name__ == '__main__':
    main()
