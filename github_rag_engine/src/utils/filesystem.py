
import os
import subprocess
from pathlib import Path
from typing import List
from src import config

def clone_repo(url: str) -> Path:
    repo_name = url.split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
        
    local_path = config.REPO_DIR / repo_name
    
    if local_path.exists():
        return local_path
    
    print(f"⏳ Cloning {url}...")
    try:
        subprocess.run(["git", "clone", url, str(local_path)], check=True)
        print("✅ Clone successful")
    except subprocess.CalledProcessError as e:
        print(f"❌ Clone failed: {e}")
        raise e
        
    return local_path

def get_source_files(directory: Path) -> List[Path]:
    source_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for file in files:
            if file.startswith("."):
                continue
                
            ext = os.path.splitext(file)[1].lower()
            if ext in config.IGNORED_EXTENSIONS:
                continue
            
            full_path = Path(root) / file
            source_files.append(full_path)
            
    return source_files

def generate_repo_map(root_dir: Path) -> str:
    lines = []
    root_path = Path(root_dir)

    for root, dirs, files in os.walk(root_path):
        if ".git" in dirs:
            dirs.remove(".git")

        for f in files:
            rel_path = Path(root, f).relative_to(root_path)
            ext = rel_path.suffix.lower()

            marker = ""
            if f.startswith(".") or ext in config.IGNORED_EXTENSIONS:
                marker = " [IGN]"

            lines.append(f"{rel_path.as_posix()}{marker}")

    return "\n".join(sorted(lines))

def get_repo_url(repo_path: Path) -> str:
    """Extracts the remote origin URL from a local git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return ""
