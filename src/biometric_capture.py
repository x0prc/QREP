from pynput import keyboard, mouse
import time
import json

class BiometricCapture:
    def __init__(self):
        self.keystroke_timings = []
        self.mouse_trajectory = []
        self.last_time = time.time()

    def on_press(self, key):
        current_time = time.time()
        self.keystroke_timings.append(current_time - self.last_time)
        self.last_time = current_time

    def on_move(self, x, y):
        self.mouse_trajectory.append((x, y, time.time()))

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.mouse_trajectory.append((x, y, time.time(), button))

    def capture(self, duration=10):
        with keyboard.Listener(on_press=self.on_press) as k_listener:
            with mouse.Listener(on_move=self.on_move, on_click=self.on_click) as m_listener:
                time.sleep(duration)
                k_listener.stop()
                m_listener.stop()
        return {
            "keystroke": self.keystroke_timings,
            "mouse": self.mouse_trajectory
        }
