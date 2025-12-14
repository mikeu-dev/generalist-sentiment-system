import json
import os
import time
import fcntl
from datetime import datetime
import redis

class TrainingStateManager:
    def __init__(self, state_file='training_state.json', upload_folder='uploads', redis_url='redis://localhost:6379/0'):
        self.state_file = os.path.join(upload_folder, state_file)
        self.redis_client = None
        
        # Try connecting to Redis
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis not available, falling back to file: {e}")
            self.redis_client = None

        # Ensure initial state exists
        if not self.get_status():
            self.reset_state()

    def _get_redis_key(self):
        return "training_state"

    def _load_state(self):
        if self.redis_client:
            try:
                data = self.redis_client.get(self._get_redis_key())
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"Redis load error: {e}")
        
        # File fallback
        try:
            if not os.path.exists(self.state_file):
                return None # Signal to reset
            
            with open(self.state_file, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            return data
        except Exception as e:
            print(f"Error loading state: {e}")
            return None

    def _save_state(self, state):
        if self.redis_client:
            try:
                self.redis_client.set(self._get_redis_key(), json.dumps(state))
                # Also save to file for persistence/debug backup
            except Exception as e:
                print(f"Redis save error: {e}")

        # Always save to file as backup/fallback
        try:
            with open(self.state_file, 'w') as f:
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
        state = self._load_state()
        if state is None:
            return self.reset_state()
        return state

    def update_status(self, progress=None, message=None, is_training=None, result=None):
        state = self.get_status()
        
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

