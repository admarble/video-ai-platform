from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigWatcher(FileSystemEventHandler):
    """Watches for configuration file changes"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if event.src_path.endswith('.yaml'):
            self.config_manager.reload_config()

class ConfigObserver:
    """Manages file system observation for config files"""
    
    def __init__(self, path: str, callback):
        self.observer = Observer()
        self.watcher = ConfigWatcher(callback)
        self.observer.schedule(self.watcher, path, recursive=False)
        
    def start(self):
        """Start watching for changes"""
        self.observer.start()
        
    def stop(self):
        """Stop watching for changes"""
        self.observer.stop()
        self.observer.join() 