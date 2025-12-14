import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing
import time
import os
import shutil
from modules.training_state import TrainingStateManager

TEST_UPLOAD_FOLDER = "test_uploads"

def worker_updater(name):
    """Simulates a worker updating the state"""
    print(f"[{name}] Starting...")
    manager = TrainingStateManager(upload_folder=TEST_UPLOAD_FOLDER)
    for i in range(1, 4):
        time.sleep(1)
        manager.update_status(progress=i*10, message=f"Updated by {name} step {i}")
        print(f"[{name}] Updated state to {i*10}%")
    
    manager.finish_training({"success": True, "worker": name})
    print(f"[{name}] Finished.")

def worker_reader(name):
    """Simulates a worker reading the state"""
    print(f"[{name}] Starting monitoring...")
    manager = TrainingStateManager(upload_folder=TEST_UPLOAD_FOLDER)
    for i in range(5):
        time.sleep(0.8)
        state = manager.get_status()
        print(f"[{name}] Read state: {state['progress']}% - {state['message']}")

if __name__ == "__main__":
    if os.path.exists(TEST_UPLOAD_FOLDER):
        shutil.rmtree(TEST_UPLOAD_FOLDER)
    os.makedirs(TEST_UPLOAD_FOLDER)

    # Initialize
    manager = TrainingStateManager(upload_folder=TEST_UPLOAD_FOLDER)
    manager.reset_state()
    
    # Start processes
    p1 = multiprocessing.Process(target=worker_updater, args=("Worker-Writer",))
    p2 = multiprocessing.Process(target=worker_reader, args=("Worker-Reader",))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    # Final check
    final_state = manager.get_status()
    print("Final State:", final_state)
    
    # Cleanup
    shutil.rmtree(TEST_UPLOAD_FOLDER)
    print("Test Complete.")
