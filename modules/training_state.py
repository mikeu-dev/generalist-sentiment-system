import json
import os
import time
import fcntl
from datetime import datetime

class TrainingStateManager:
    def __init__(self, state_file='training_state.json', upload_folder='uploads'):
        self.state_file = os.path.join(upload_folder, state_file)
        # Ensure initial state exists if not present
        if not os.path.exists(self.state_file):
            self.reset_state()

    def _load_state(self):
        try:
            if not os.path.exists(self.state_file):
                return self.reset_state()
            
            with open(self.state_file, 'r') as f:
                # Simple shared lock for reading
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            return data
        except Exception as e:
            # Fallback if file is corrupt
            print(f"Error loading state: {e}")
            return self.reset_state()

    def _save_state(self, state):
        try:
            with open(self.state_file, 'w') as f:
                # Exclusive lock for writing
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(state, f)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Error saving state: {e}")

    def reset_state(self):
        state = {
            "is_training": False,
            "progress": 0,
            "message": "Idle",
            "result": None,
            "last_updated": datetime.now().isoformat()
        }
        self._save_state(state)
        return state

    def get_status(self):
        return self._load_state()

    def update_status(self, progress=None, message=None, is_training=None, result=None):
        state = self._load_state()
        
        if progress is not None:
            state['progress'] = progress
        if message is not None:
            state['message'] = message
        if is_training is not None:
            state['is_training'] = is_training
        if result is not None:
            state['result'] = result
            
        state['last_updated'] = datetime.now().isoformat()
        self._save_state(state)
        
    def start_training(self):
        self.update_status(is_training=True, progress=0, message="Memulai training...", result=None)

    def finish_training(self, result):
        self.update_status(is_training=False, progress=100, message="Training selesai!", result=result)
        
    def error_training(self, error_msg):
        result = {"success": False, "error": error_msg}
        self.update_status(is_training=False, progress=0, message=f"Error: {error_msg}", result=result)

