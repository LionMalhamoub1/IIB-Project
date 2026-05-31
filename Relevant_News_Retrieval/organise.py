import os
import re
import shutil
from pathlib import Path

def organize_to_day_level(base_dir: str):
    base_path = Path(base_dir)
    
    # We use a list comprehension with rglob for multi-extension support
    extensions = ['*.csv', '*.txt']
    all_files = []
    for ext in extensions:
        all_files.extend(list(base_path.rglob(ext)))
    
    print(f"Scanning {base_path}... Found {len(all_files)} total files.")
    
    moved_count = 0
    for file_path in all_files:
        # 2. Extract the YYYYMMDD date from the filename
        date_match = re.search(r"(\d{8})", file_path.name)
        if not date_match:
            continue
            
        date_str = date_match.group(1)
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        
        # 3. Define the target directory: Year/Month/Day
        target_dir = base_path / year / month / day
        
        # 4. Check if we actually need to move it
        if file_path.parent.resolve() != target_dir.resolve():
            target_dir.mkdir(parents=True, exist_ok=True)
            destination = target_dir / file_path.name
            
            # Handle potential filename collisions
            if not destination.exists():
                shutil.move(str(file_path), str(destination))
                print(f"Moved: {file_path.name} -> {year}/{month}/{day}/")
                moved_count += 1
            else:
                print(f"Skipped: {file_path.name} (already exists in destination)")

    print(f"\nTask Complete. Moved {moved_count} files.")

if __name__ == "__main__":
    TARGET = "data/interim/_state"
    organize_to_day_level(TARGET)