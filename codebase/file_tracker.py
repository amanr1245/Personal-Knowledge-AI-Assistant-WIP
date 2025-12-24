"""
File Tracker Module

Track file changes for incremental updates.
Computes content hashes and detects new, modified, unchanged, and deleted files.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileStatus:
    """Status of a file for incremental processing."""
    path: str
    content_hash: str
    mtime: float
    status: str  # 'new', 'modified', 'unchanged', 'deleted'

    @property
    def mtime_datetime(self) -> datetime:
        """Get mtime as datetime object."""
        return datetime.fromtimestamp(self.mtime)


def compute_file_hash(path: str) -> str:
    """
    Compute SHA256 hash of file content.

    Args:
        path: Path to the file

    Returns:
        First 16 characters of SHA256 hash
    """
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def get_file_status(path: str, db_hash: Optional[str] = None) -> FileStatus:
    """
    Check if file is new, modified, or unchanged.

    Args:
        path: Path to the file
        db_hash: Hash stored in database (None if file not in DB)

    Returns:
        FileStatus with current hash, mtime, and status
    """
    current_hash = compute_file_hash(path)
    mtime = os.path.getmtime(path)

    if db_hash is None:
        status = 'new'
    elif current_hash != db_hash:
        status = 'modified'
    else:
        status = 'unchanged'

    return FileStatus(path, current_hash, mtime, status)


def scan_directory(directory: str, extensions: Set[str]) -> List[str]:
    """
    Scan directory for supported files.

    Args:
        directory: Directory path to scan
        extensions: Set of supported extensions (e.g., {'.pdf', '.txt'})

    Returns:
        List of absolute file paths
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in extensions:
                files.append(os.path.join(root, filename))
    return files


def categorize_files(
    current_files: List[str],
    existing_files: Dict[str, str]
) -> Dict[str, List[FileStatus]]:
    """
    Categorize files into new, modified, unchanged, and deleted.

    Args:
        current_files: List of current file paths on disk
        existing_files: Dict mapping path -> content_hash from database

    Returns:
        Dict with keys 'new', 'modified', 'unchanged', 'deleted'
        containing lists of FileStatus objects
    """
    result = {
        'new': [],
        'modified': [],
        'unchanged': [],
        'deleted': []
    }

    current_set = set(current_files)
    existing_set = set(existing_files.keys())

    # Check each current file
    for path in current_files:
        db_hash = existing_files.get(path)
        status = get_file_status(path, db_hash)
        result[status.status].append(status)

    # Find deleted files
    deleted_paths = existing_set - current_set
    for path in deleted_paths:
        result['deleted'].append(FileStatus(
            path=path,
            content_hash=existing_files[path],
            mtime=0,
            status='deleted'
        ))

    return result


def get_files_to_process(
    current_files: List[str],
    existing_files: Dict[str, str]
) -> tuple:
    """
    Get lists of files that need processing and deletion.

    Args:
        current_files: List of current file paths on disk
        existing_files: Dict mapping path -> content_hash from database

    Returns:
        (to_process, to_delete) tuple of lists
    """
    categorized = categorize_files(current_files, existing_files)

    to_process = [fs.path for fs in categorized['new'] + categorized['modified']]
    to_delete = [fs.path for fs in categorized['deleted']]

    # Also delete modified files (they'll be re-processed)
    to_delete_first = [fs.path for fs in categorized['modified']]

    return to_process, to_delete, to_delete_first


def print_file_status_summary(categorized: Dict[str, List[FileStatus]]) -> None:
    """Print a summary of file statuses."""
    print(f"\nFile Status Summary:")
    print(f"  New files:       {len(categorized['new'])}")
    print(f"  Modified files:  {len(categorized['modified'])}")
    print(f"  Unchanged files: {len(categorized['unchanged'])}")
    print(f"  Deleted files:   {len(categorized['deleted'])}")

    if categorized['new']:
        print(f"\nNew files:")
        for fs in categorized['new'][:5]:
            print(f"    + {fs.path}")
        if len(categorized['new']) > 5:
            print(f"    ... and {len(categorized['new']) - 5} more")

    if categorized['modified']:
        print(f"\nModified files:")
        for fs in categorized['modified'][:5]:
            print(f"    ~ {fs.path}")
        if len(categorized['modified']) > 5:
            print(f"    ... and {len(categorized['modified']) - 5} more")

    if categorized['deleted']:
        print(f"\nDeleted files:")
        for fs in categorized['deleted'][:5]:
            print(f"    - {fs.path}")
        if len(categorized['deleted']) > 5:
            print(f"    ... and {len(categorized['deleted']) - 5} more")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python file_tracker.py <directory> [extensions]")
        print("  extensions: comma-separated list (default: .pdf,.txt,.md)")
        sys.exit(1)

    directory = sys.argv[1]
    ext_str = sys.argv[2] if len(sys.argv) > 2 else ".pdf,.txt,.md"
    extensions = set(ext_str.split(','))

    print(f"Scanning {directory} for {extensions}...")
    files = scan_directory(directory, extensions)

    print(f"\nFound {len(files)} files:")
    for f in files[:10]:
        hash_val = compute_file_hash(f)
        print(f"  {hash_val} {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
