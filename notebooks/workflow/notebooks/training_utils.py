import time
from tensorflow.keras.callbacks import Callback

class TimeLimitCallback(Callback):
    """Stop training if it exceeds a specified time limit (in seconds)."""
    def __init__(self, max_duration_seconds):
        super().__init__()
        self.max_duration = max_duration_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_duration:
            print(f"\n\nTraining stopped after {elapsed:.2f}s (limit: {self.max_duration}s)")
            self.model.stop_training = True

