#!/usr/bin/env python3
"""
File sync watcher - watches local project and syncs changes to remote server via SCP/paramiko
"""
import time
import os
import sys
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import paramiko
from scp import SCPClient

LOCAL_PATH = Path(r"H:\PanelCleaner-CUDA-build-masterV")
REMOTE_HOST = "YOUR_SERVER_IP"
REMOTE_PORT = 56511
REMOTE_USER = "root"
REMOTE_PATH = "/workspace/PanelCleaner-CUDA-build-masterV"
SSH_KEY = r"C:\Users\National\.ssh\id_rsa"

EXCLUDE_PATTERNS = [
    "__pycache__", "*.pyc", "*.pyo", "*.pyd",
    ".git", ".gitignore",
    ".env", "*.env",
    "logs/*.log", "logs/panelcleaner.log",
    "temp_chapters", "credentials",
    "*.vhdx", "DockerDesktopWSL",
    "New Text Document.txt",
    "$RECYCLE.BIN",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules",
    "*.egg-info", "dist", "build",
    ".DS_Store", "Thumbs.db",
    "sync_to_server.py",
    "bubble_embeddings_cache.json",
]

def should_exclude(path: Path) -> bool:
    rel = path.relative_to(LOCAL_PATH) if path.is_relative_to(LOCAL_PATH) else path
    rel_str = str(rel).replace("\\", "/")
    for pattern in EXCLUDE_PATTERNS:
        if "*" in pattern:
            import fnmatch
            if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
        elif pattern in rel_str or pattern == path.name:
            return True
    return False

class SyncManager:
    def __init__(self):
        self.ssh = None
        self.scp = None
        self.connect()
    
    def connect(self):
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                REMOTE_HOST, 
                port=REMOTE_PORT, 
                username=REMOTE_USER, 
                key_filename=SSH_KEY,
                timeout=30
            )
            self.scp = SCPClient(self.ssh.get_transport())
            print(f"[{time.strftime('%H:%M:%S')}] Connected to server")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Connection failed: {e}")
            raise
    
    def ensure_remote_dir(self, remote_dir):
        stdin, stdout, stderr = self.ssh.exec_command(f"mkdir -p '{remote_dir}'")
        stdout.channel.recv_exit_status()
    
    def upload_file(self, local_path: Path):
        if should_exclude(local_path):
            return
        try:
            rel = local_path.relative_to(LOCAL_PATH)
            remote_file = f"{REMOTE_PATH}/{rel}".replace("\\", "/")
            remote_dir = "/".join(remote_file.split("/")[:-1])
            self.ensure_remote_dir(remote_dir)
            self.scp.put(str(local_path), remote_file)
            print(f"[{time.strftime('%H:%M:%S')}] UPLOAD: {rel}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] UPLOAD FAILED {local_path}: {e}")
    
    def delete_remote(self, local_path: Path):
        if should_exclude(local_path):
            return
        try:
            rel = local_path.relative_to(LOCAL_PATH)
            remote_file = f"{REMOTE_PATH}/{rel}".replace("\\", "/")
            stdin, stdout, stderr = self.ssh.exec_command(f"rm -f '{remote_file}'")
            stdout.channel.recv_exit_status()
            print(f"[{time.strftime('%H:%M:%S')}] DELETE: {rel}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] DELETE FAILED {local_path}: {e}")
    
    def initial_sync(self):
        print(f"[{time.strftime('%H:%M:%S')}] Starting initial sync...")
        count = 0
        for root, dirs, files in os.walk(LOCAL_PATH):
            root_path = Path(root)
            # Skip excluded dirs
            dirs[:] = [d for d in dirs if not should_exclude(root_path / d)]
            
            for f in files:
                fpath = root_path / f
                if not should_exclude(fpath):
                    self.upload_file(fpath)
                    count += 1
        print(f"[{time.strftime('%H:%M:%S')}] Initial sync complete: {count} files")

    def close(self):
        if self.scp:
            self.scp.close()
        if self.ssh:
            self.ssh.close()

class SyncHandler(FileSystemEventHandler):
    def __init__(self, sync_manager):
        self.sync = sync_manager
        self.pending = set()
        self.lock = threading.Lock()
        self.last_sync = 0
    
    def on_any_event(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if should_exclude(path):
            return
        with self.lock:
            self.pending.add(path)
    
    def process_pending(self):
        with self.lock:
            if not self.pending:
                return
            if time.time() - self.last_sync < 1:
                return
            files = list(self.pending)
            self.pending.clear()
            self.last_sync = time.time()
        
        for f in files:
            if f.exists():
                self.sync.upload_file(f)
            else:
                self.sync.delete_remote(f)

def main():
    sync = SyncManager()
    try:
        sync.initial_sync()
    except Exception as e:
        print(f"Initial sync failed: {e}")
        sync.close()
        return 1
    
    handler = SyncHandler(sync)
    observer = Observer()
    observer.schedule(handler, str(LOCAL_PATH), recursive=True)
    observer.start()
    
    print(f"[{time.strftime('%H:%M:%S')}] Watching {LOCAL_PATH} for changes...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            handler.process_pending()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        observer.stop()
        observer.join()
        sync.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())