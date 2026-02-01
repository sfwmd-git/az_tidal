#!/usr/bin/env python3

import os

CONFLICT_FILES = {"requests.py", "requests.pyc", "cdsapi.py", "cdsapi.pyc"}

def find_conflicting_files(start_dir=None):
    """
    Search the current directory (or a given start_dir) and all its
    parent directories for potential file name conflicts that overshadow
    standard libraries: requests.py, requests.pyc, cdsapi.py, cdsapi.pyc.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    
    visited = set()
    current_dir = os.path.abspath(start_dir)

    while True:
        if current_dir in visited:
            # Safety check in case of a symlink loop
            break
        visited.add(current_dir)

        # List all files in the current directory
        try:
            all_files = os.listdir(current_dir)
        except PermissionError:
            # If we don't have permission to list files, just skip
            pass
        else:
            # Check for conflicts
            conflicts = [f for f in all_files if f in CONFLICT_FILES]
            if conflicts:
                print(f"Found potential conflicts in: {current_dir}")
                for c in conflicts:
                    full_path = os.path.join(current_dir, c)
                    print(f"  - {full_path}")

        # Move up to the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if parent_dir == current_dir:
            # We've reached root
            break
        current_dir = parent_dir


if __name__ == "__main__":
    print("Checking for local files named requests.py, requests.pyc, cdsapi.py, cdsapi.pyc...")
    find_conflicting_files()
    print("Done.")
