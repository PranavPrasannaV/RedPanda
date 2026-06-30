"""
Syzygy Tablebase Downloader

Downloads 6-piece Syzygy tablebases from Lichess servers.
These provide perfect endgame play for positions with 6 or fewer pieces.

Usage:
    python download_tablebases.py

The script will:
1. Scrape the file listing from Lichess
2. Download all .rtbw (WDL) and .rtbz (DTZ) files
3. Save them to the ./syzygy folder
4. Resume from where it left off if interrupted
"""

import os
import re
import time
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_URL = "https://tablebase.lichess.ovh/tables/standard"
DOWNLOAD_DIR = Path(__file__).parent / "syzygy"

# What to download - set to True to enable
DOWNLOAD_CONFIG = {
    "3-4-5-wdl": True,   # ~1 GB - Basic tablebases WDL (3-5 pieces)
    "3-4-5-dtz": True,   # ~200 MB - Basic tablebases DTZ (3-5 pieces)
    "6-wdl": True,       # ~68 GB - 6-piece WDL (Win/Draw/Loss)
    "6-dtz": False,      # ~82 GB - 6-piece DTZ (Distance to Zero) - SKIP to save space
}

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks


def get_file_list(url):
    """Scrape the directory listing to get all tablebase files."""
    print(f"Fetching file list from {url}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
    
    # Parse HTML for file links
    pattern = r'href="([^"]+\.rtb[wz])"'
    files = re.findall(pattern, response.text)
    
    # Build full URLs
    file_urls = []
    for filename in files:
        if not filename.startswith("http"):
            file_url = f"{url.rstrip('/')}/{filename}"
        else:
            file_url = filename
        file_urls.append((filename, file_url))
    
    return file_urls


def download_file(filename, url, dest_dir):
    """Download a single file with progress bar and resume support."""
    dest_path = dest_dir / filename
    
    # Check if already downloaded
    if dest_path.exists():
        try:
            response = requests.head(url, timeout=30)
            remote_size = int(response.headers.get('content-length', 0))
            local_size = dest_path.stat().st_size
            
            if remote_size > 0 and local_size == remote_size:
                return "SKIP", local_size
        except:
            pass
    
    # Download with retries
    for attempt in range(MAX_RETRIES):
        try:
            # Use stream=True and no timeout for iter_content
            # Only timeout on initial connection
            response = requests.get(url, stream=True, timeout=(30, None))
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Progress bar for this file
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=filename[:30], leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return "OK", total_size
            
        except KeyboardInterrupt:
            # Clean up partial file on interrupt
            if dest_path.exists():
                dest_path.unlink()
            raise
            
        except Exception as e:
            print(f"\n  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                # Clean up partial file on failure
                if dest_path.exists():
                    dest_path.unlink()
                return "FAIL", 0
    
    return "FAIL", 0


def download_category(category, base_url, dest_dir):
    """Download all files from a category sequentially."""
    url = f"{base_url}/{category}/"
    
    print(f"\n{'='*60}")
    print(f"Downloading: {category}")
    print(f"URL: {url}")
    print(f"Destination: {dest_dir}")
    print('='*60)
    
    files = get_file_list(url)
    
    if not files:
        print(f"No files found in {category}")
        return
    
    print(f"Found {len(files)} files to download\n")
    
    # Sequential download with overall progress
    ok_count = 0
    skip_count = 0
    fail_count = 0
    total_bytes = 0
    
    for i, (filename, file_url) in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {filename}")
        
        status, size = download_file(filename, file_url, dest_dir)
        
        if status == "OK":
            ok_count += 1
            total_bytes += size
            size_mb = size / (1024 * 1024)
            print(f"  [OK] Downloaded ({size_mb:.1f} MB)")
        elif status == "SKIP":
            skip_count += 1
            print(f"  [.] Already exists, skipped")
        else:
            fail_count += 1
            print(f"  [x] Failed")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Results for {category}:")
    print(f"  Downloaded: {ok_count} ({total_bytes / (1024**3):.2f} GB)")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")
    print('='*60)


def estimate_download_size():
    """Estimate total download size based on configuration."""
    sizes = {
        "3-4-5-wdl": 1.0,   # ~1 GB
        "3-4-5-dtz": 0.2,   # ~200 MB
        "6-wdl": 68.0,      # ~68 GB
        "6-dtz": 82.0,      # ~82 GB
    }
    
    total_gb = sum(sizes[cat] for cat, enabled in DOWNLOAD_CONFIG.items() if enabled)
    return total_gb


def main():
    print("="*60)
    print("Syzygy Tablebase Downloader")
    print("="*60)
    
    # Create destination directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownload directory: {DOWNLOAD_DIR}")
    
    # Estimate size
    total_gb = estimate_download_size()
    print(f"Estimated download size: {total_gb:.1f} GB")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(DOWNLOAD_DIR)
        free_gb = free / (1024**3)
        print(f"Available disk space: {free_gb:.1f} GB")
        
        if free_gb < total_gb * 1.1:
            print(f"\nWARNING: You may not have enough disk space!")
            print(f"Required: ~{total_gb:.1f} GB, Available: {free_gb:.1f} GB")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    except:
        pass
    
    # Show configuration
    print("\nDownload configuration:")
    for category, enabled in DOWNLOAD_CONFIG.items():
        status = "ENABLED" if enabled else "disabled"
        print(f"  {category}: {status}")
    
    print("\nStarting download (sequential mode for stability)...")
    print("(Press Ctrl+C to stop - progress is saved)\n")
    
    try:
        # Download each enabled category
        for category, enabled in DOWNLOAD_CONFIG.items():
            if enabled:
                download_category(category, BASE_URL, DOWNLOAD_DIR)
        
        print("\n" + "="*60)
        print("Download complete!")
        print(f"Tablebases saved to: {DOWNLOAD_DIR}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Run the script again to resume from where you left off.")


if __name__ == "__main__":
    main()

