"""
Hand Sign Recognition - Real-time Inference (SENTENCE BUILDER)
---------------------------------------------------------------
SENTENCE BUILDER FEATURES:
1. Single SPACE gesture: Adds space to buffer (finishes word)
2. Double SPACE gesture (consecutive): Speaks the complete sentence
3. TTS works correctly for multiple words (engine reinit per word)
4. Better cooldown logic to prevent duplicate triggers
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import queue
import time
import pickle
import os
import urllib.request
from collections import deque


class HandSignRecognizer:
    def __init__(self,
                 model_path='model.tflite',
                 label_encoder_path='label_encoder.pkl',
                 confidence_threshold=0.75,
                 debounce_time=1.5):

        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.confidence_threshold = confidence_threshold
        self.debounce_time = debounce_time

        # Download MediaPipe model if needed
        self.mediapipe_model = 'hand_landmarker.task'
        self._download_mediapipe_model()

        # Initialize MediaPipe
        self._init_mediapipe()

        # Load TFLite model
        self._load_tflite_model()

        # Load label encoder
        self._load_label_encoder()

        # TTS setup
        self._init_tts()

        # State variables
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_spoken_gesture = None
        self.gesture_confirmed = False
        self.last_confirmed_time = None
        self.confirmed_cooldown = 2.0
        self.last_added_letter = None

        # SPACE gesture specific (Sentence Builder)
        self.space_cooldown = 2.0  # Prevent rapid SPACE triggering
        self.last_space_time = 0
        self.space_sequence_count = 0  # Track consecutive SPACE gestures
        self.space_sequence_timeout = 3.0  # Time window for consecutive SPACE detection
        self.last_space_action_time = 0  # When last SPACE action occurred

        # Word building (now sentence building)
        self.word_buffer = []  # Contains letters and spaces, forms sentence
        self.last_letter_time = None
        self.last_letter_added_time = 0

        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.gesture_history = deque(maxlen=10)

        # UI state
        self.show_help = False
        self.recent_letter_flash_time = 0
        self.recent_letter = None

        print("✓ Hand Sign Recognizer initialized")

    def _download_mediapipe_model(self):
        """Download MediaPipe model if not present"""
        if not os.path.exists(self.mediapipe_model):
            print("📥 Downloading MediaPipe model...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, self.mediapipe_model)
            print("✓ Model downloaded")

    def _init_mediapipe(self):
        """Initialize MediaPipe Hand Landmarker"""
        print("🔧 Initializing MediaPipe...")
        base_options = python.BaseOptions(model_asset_path=self.mediapipe_model)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    def _load_tflite_model(self):
        """Load TFLite model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"📦 Loading TFLite model: {self.model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"✓ Model loaded (input shape: {self.input_details[0]['shape']})")

    def _load_label_encoder(self):
        """Load label encoder"""
        if not os.path.exists(self.label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {self.label_encoder_path}")

        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        print(f"✓ Loaded {len(self.label_encoder.classes_)} classes: {list(self.label_encoder.classes_)}")

    def _init_tts(self):
        """Initialize Text-to-Speech queue"""
        self.tts_queue = queue.Queue()
        self.tts_busy = threading.Event()

        # Start TTS worker thread
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        print("✓ TTS initialized")

    def _tts_worker(self):
        """
        FIXED TTS Worker Thread
        - Reinitializes engine for EACH word to prevent hanging
        - This solves the "only first word speaks" problem
        """
        while True:
            try:
                text = self.tts_queue.get(timeout=0.5)
                if text is None:
                    break

                self.tts_busy.set()

                # CRITICAL FIX: Reinitialize engine for each word
                # This prevents the engine from "dying" after first use
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)

                    print(f"🔊 TTS: Speaking '{text}'...")
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()

                    # Clean up engine reference
                    del engine

                except Exception as e:
                    print(f"❌ TTS Error: {e}")

                self.tts_busy.clear()
                self.tts_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ TTS Worker Error: {e}")
                self.tts_busy.clear()

    def speak(self, text):
        """Add text to TTS queue (non-blocking)"""
        if text and len(text.strip()) > 0:
            self.tts_queue.put(text.strip())
            return True
        return False

    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize 42 features (X, Y coordinates only)"""
        landmarks = []
        for lm in hand_landmarks[:21]:
            landmarks.append([lm.x, lm.y])

        while len(landmarks) < 21:
            landmarks.append([0.0, 0.0])

        landmarks = np.array(landmarks)
        wrist = landmarks[0]
        normalized = landmarks - wrist

        x_coords = normalized[:, 0]
        y_coords = normalized[:, 1]

        return np.concatenate([x_coords, y_coords])

    def predict(self, landmarks):
        """Run inference and return (class, confidence)"""
        input_data = np.array([landmarks], dtype=np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_idx = np.argmax(output_data[0])
        confidence = output_data[0][predicted_idx]

        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]

        if isinstance(predicted_class, (np.integer, np.floating)):
            predicted_class = str(predicted_class)
        elif not isinstance(predicted_class, str):
            predicted_class = str(predicted_class)

        return predicted_class, confidence

    def handle_gesture(self, gesture, confidence):
        """
        Sentence Builder Gesture State Machine
        - Single SPACE: Adds space to buffer (finishes word)
        - Double SPACE (consecutive): Speaks the complete sentence
        - Letters and words: Added to buffer
        - Proper cooldown prevents duplicate triggers
        """
        current_time = time.time()

        # Low confidence - ignore
        if confidence < self.confidence_threshold:
            self.current_gesture = None
            self.gesture_start_time = None
            self.gesture_confirmed = False
            return None, None

        # SPACE GESTURE HANDLING (SENTENCE BUILDER)
        if gesture == 'SPACE':
            # Check cooldown to prevent rapid triggering
            time_since_last_space = current_time - self.last_space_time

            if time_since_last_space < self.space_cooldown:
                # Still in cooldown, ignore
                return None, None

            # Check if this is a consecutive SPACE gesture
            time_since_last_space_action = current_time - self.last_space_action_time
            if time_since_last_space_action < self.space_sequence_timeout:
                # Consecutive SPACE detected
                self.space_sequence_count += 1
            else:
                # New SPACE sequence (first SPACE)
                self.space_sequence_count = 1

            self.last_space_time = current_time

            # Single SPACE: Add space to buffer (finish word)
            if self.space_sequence_count == 1:
                if self.word_buffer:  # Only add space if buffer has content
                    # Don't add space if last element is already a space
                    if self.word_buffer[-1] != ' ':
                        self.word_buffer.append(' ')
                        self.last_letter_time = current_time
                        self.last_space_action_time = current_time
                        sentence_text = ''.join(self.word_buffer)
                        print(f"✓ Space added | Sentence: '{sentence_text}'")
                        self.gesture_confirmed = False
                        self.last_confirmed_time = None
                        return 'add_space', ' '
                    else:
                        # Last element is space, but this is first SPACE - ignore duplicate
                        return None, None
                else:
                    print("⚠️ SPACE detected but buffer is empty (add letters first)")
                    self.current_gesture = None
                    self.gesture_start_time = None
                    self.gesture_confirmed = False
                    return None, None

            # Double SPACE (2 consecutive): Speak the sentence
            elif self.space_sequence_count == 2:
                if self.word_buffer and len(self.word_buffer) > 0:
                    sentence = ''.join(self.word_buffer).strip()  # Remove trailing spaces
                    if sentence:
                        print(f"✅ DOUBLE SPACE detected! Speaking sentence: '{sentence}'")
                        
                        # Clear buffer and reset states
                        self.word_buffer.clear()
                        self.last_letter_time = None
                        self.last_added_letter = None
                        self.last_space_action_time = current_time
                        self.space_sequence_count = 0  # Reset counter

                        # Reset gesture tracking
                        self.current_gesture = None
                        self.gesture_start_time = None
                        self.gesture_confirmed = False
                        self.last_confirmed_time = None

                        return 'speak_sentence', sentence
                    else:
                        print("⚠️ DOUBLE SPACE detected but sentence is empty")
                        self.space_sequence_count = 0
                        return None, None
                else:
                    print("⚠️ DOUBLE SPACE detected but buffer is empty")
                    self.space_sequence_count = 0
                    self.current_gesture = None
                    self.gesture_start_time = None
                    self.gesture_confirmed = False
                    return None, None

            # Triple or more SPACE: Reset and treat as new sequence
            else:
                # Reset and start over (ignore extra SPACEs)
                self.space_sequence_count = 0
                return None, None

        # Same gesture detection (for letters)
        if gesture == self.current_gesture:
            if self.gesture_start_time:
                hold_time = current_time - self.gesture_start_time

                # Reset confirmation if cooldown passed
                if self.gesture_confirmed and self.last_confirmed_time:
                    time_since_confirmed = current_time - self.last_confirmed_time
                    if time_since_confirmed >= self.confirmed_cooldown:
                        self.gesture_confirmed = False

                # Gesture confirmed after debounce
                if hold_time >= self.debounce_time and not self.gesture_confirmed:
                    self.gesture_confirmed = True
                    self.last_confirmed_time = current_time

                    # Multi-character words (add to buffer, don't speak)
                    if isinstance(gesture, str) and len(gesture) > 1:
                        self.word_buffer.append(gesture)
                        self.last_letter_time = current_time
                        self.last_added_letter = gesture
                        self.last_letter_added_time = current_time
                        self.gesture_confirmed = False
                        self.last_confirmed_time = None
                        return 'add_letter', gesture

                    # Single letters
                    else:
                        time_since_last_add = current_time - self.last_letter_added_time
                        can_add_duplicate = (gesture != self.last_added_letter) or (time_since_last_add > 0.5)

                        if can_add_duplicate:
                            self.word_buffer.append(gesture)
                            self.last_letter_time = current_time
                            self.last_added_letter = gesture
                            self.last_letter_added_time = current_time
                            self.recent_letter = gesture
                            self.recent_letter_flash_time = current_time
                            self.gesture_confirmed = False
                            self.last_confirmed_time = None
                            return 'add_letter', gesture

        # New gesture detected (not SPACE, and not same as current)
        else:
            self.current_gesture = gesture
            self.gesture_start_time = current_time
            self.gesture_confirmed = False
            self.last_confirmed_time = None
            if gesture != self.last_added_letter:
                self.last_added_letter = None
            
            # Reset SPACE sequence counter when non-SPACE gesture detected
            if gesture != 'SPACE':
                time_since_last_space_action = current_time - self.last_space_action_time
                if time_since_last_space_action > self.space_sequence_timeout:
                    self.space_sequence_count = 0

        return None, None

    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks"""
        h, w, _ = frame.shape

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    def draw_ui(self, frame, gesture, confidence, fps):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        current_time = time.time()

        # Help overlay
        if self.show_help:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            help_lines = [
                "CONTROLS:",
                "  Q - Quit",
                "  C - Clear sentence buffer",
                "  B - Backspace (remove last letter/space)",
                "  H - Toggle this help",
                "",
                "SENTENCE BUILDER:",
                "  1. Hold gesture for 1.5s to add letter",
                "  2. Letters accumulate in sentence buffer",
                "  3. 1 SPACE = Add space (finish word)",
                "  4. 2 SPACE (consecutive) = Speak sentence",
                "  5. For duplicate letters (HELLO):",
                "     - Add H, E, L",
                "     - Hold L again for second L",
                "     - Add O, then SPACE for space",
                "     - Add next word, then 2 SPACE to speak"
            ]

            start_y = h // 2 - (len(help_lines) * 22) // 2
            for i, line in enumerate(help_lines):
                y_pos = start_y + i * 22
                if line.endswith(':'):
                    color = (255, 255, 0)
                    weight = 2
                elif line.startswith('  '):
                    color = (255, 255, 255)
                    weight = 1
                else:
                    color = (200, 200, 200)
                    weight = 1

                cv2.putText(frame, line, (w // 2 - 250, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)
            return

        # Semi-transparent top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        y_offset = 30

        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Current gesture
        if gesture:
            if confidence >= self.confidence_threshold:
                color = (0, 255, 0)
                status = "DETECTING"
            else:
                color = (0, 165, 255)
                status = "LOW CONF"

            # Progress bar for hold timer
            if self.gesture_start_time and gesture == self.current_gesture:
                hold_time = current_time - self.gesture_start_time
                progress = min(hold_time / self.debounce_time, 1.0)
                bar_width = int(300 * progress)

                cv2.rectangle(frame, (10, y_offset + 35), (310, y_offset + 50), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, y_offset + 35), (10 + bar_width, y_offset + 50), color, -1)

                cv2.putText(frame, f"{int(progress * 100)}%", (320, y_offset + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            text = f"{status}: {gesture} ({confidence:.2f})"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Special hint for SPACE gesture (Sentence Builder)
            if gesture == 'SPACE':
                if self.word_buffer:
                    sentence_preview = ''.join(self.word_buffer)
                    if self.space_sequence_count == 0 or self.space_sequence_count == 1:
                        hint = f"1 SPACE: Add space | 2 SPACE: Speak sentence"
                        hint_color = (0, 255, 255)
                    else:
                        hint = f"Release to speak: {sentence_preview.strip()}"
                        hint_color = (0, 255, 0)
                else:
                    hint = "Buffer empty - add letters first"
                    hint_color = (0, 165, 255)
                cv2.putText(frame, hint, (10, y_offset + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, hint_color, 2)

            # Hint for duplicate letters
            elif len(gesture) == 1 and self.word_buffer and self.word_buffer[-1] == gesture:
                hint_text = f"Hold {gesture} again for duplicate"
                cv2.putText(frame, hint_text, (10, y_offset + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        else:
            cv2.putText(frame, "NO HAND DETECTED", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Recent letter flash animation
        if self.recent_letter and (current_time - self.recent_letter_flash_time) < 0.5:
            flash_x = w - 150
            flash_y = y_offset + 50
            cv2.putText(frame, f"+{self.recent_letter}", (flash_x, flash_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Gesture history
        if self.gesture_history:
            history_text = "Recent: "+" → ".join(list(self.gesture_history)[-5:])
            cv2.putText(frame, history_text, (10, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Large sentence display (center bottom)
        if self.word_buffer:
            sentence_text = ''.join(self.word_buffer).upper()
            text_size = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 120

            padding = 30
            cv2.rectangle(frame,
                          (text_x - padding, text_y - text_size[1] - padding),
                          (text_x + text_size[0] + padding, text_y + padding),
                          (0, 0, 0), -1)
            cv2.rectangle(frame,
                          (text_x - padding, text_y - text_size[1] - padding),
                          (text_x + text_size[0] + padding, text_y + padding),
                          (255, 255, 0), 3)

            cv2.putText(frame, sentence_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4)

            # Blinking cursor
            cursor_x = text_x + text_size[0] + 10
            if int(current_time * 2) % 2:
                cv2.line(frame, (cursor_x, text_y - text_size[1]),
                         (cursor_x, text_y), (255, 255, 255), 4)

            # Instruction (Sentence Builder)
            instruction = "1 SPACE = Add space | 2 SPACE = Speak sentence"
            inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            inst_x = (w - inst_size[0]) // 2
            cv2.putText(frame, instruction, (inst_x, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Help hint
        cv2.putText(frame, "Press H for help", (w - 200, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    def run(self):
        """Main recognition loop"""
        print("\n" + "=" * 70)
        print("🎥 HAND SIGN RECOGNITION TO SPEECH (SENTENCE BUILDER)")
        print("=" * 70)
        print("HOW IT WORKS:")
        print("  1. Hold gesture for 1.5s to add letter to sentence")
        print("  2. Letters accumulate in the sentence buffer")
        print("  3. 1 SPACE = Add space (finish word)")
        print("  4. 2 SPACE (consecutive) = SPEAK the complete sentence")
        print("  5. For duplicates (HELLO): H-E-L, then L again, then O")
        print("\nEXAMPLE:")
        print("  H-E-L-L-O [SPACE] → 'HELLO '")
        print("  W-O-R-L-D [SPACE] → 'HELLO WORLD '")
        print("  [SPACE again] → Speaks: 'HELLO WORLD'")
        print("\nCONTROLS:")
        print("  Q - Quit  |  C - Clear buffer  |  B - Backspace  |  H - Help")
        print("=" * 70 + "\n")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_time_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # FPS calculation
            frame_time_end = time.time()
            fps = 1.0 / (frame_time_end - frame_time_start + 1e-6)
            frame_time_start = frame_time_end
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            # Mirror frame
            frame = cv2.flip(frame, 1)

            # RGB conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Hand detection
            detection_result = self.hand_landmarker.detect(mp_image)

            gesture = None
            confidence = 0.0

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                self.draw_landmarks(frame, hand_landmarks)

                landmarks = self.extract_landmarks(hand_landmarks)
                gesture, confidence = self.predict(landmarks)

                action, data = self.handle_gesture(gesture, confidence)

                if action == 'add_letter':
                    print(f"✓ Added: {data} | Current sentence: {''.join(self.word_buffer)}")
                    self.gesture_history.append(data)

                elif action == 'add_space':
                    self.gesture_history.append('SPACE')

                elif action == 'speak_word':
                    self.speak(data)
                    print(f"🔊 Speaking word: {data}")
                    self.gesture_history.append(f"[{data}]")

                elif action == 'speak_sentence':
                    self.speak(data)
                    print(f"🔊 Speaking sentence: {data}")
                    self.gesture_history.append(f"[{data}]")

            # Draw UI
            self.draw_ui(frame, gesture, confidence, avg_fps)

            cv2.imshow('Hand Sign Recognition', frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                self.word_buffer.clear()
                self.last_letter_time = None
                self.last_added_letter = None
                self.space_sequence_count = 0
                self.last_space_action_time = 0
                print("🗑️  Sentence buffer cleared")
            elif key == ord('b') or key == 8:
                if self.word_buffer:
                    removed = self.word_buffer.pop()
                    self.last_letter_time = time.time()
                    self.last_added_letter = None
                    print(f"⬅️  Removed '{removed}'. Word: {''.join(self.word_buffer)}")
                else:
                    print("⚠️ Buffer is empty")
            elif key == ord('h'):
                self.show_help = not self.show_help

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hand_landmarker.close()
        self.tts_queue.put(None)

        print("\n✓ Application closed")


if __name__ == "__main__":
    recognizer = HandSignRecognizer(
        confidence_threshold=0.75,
        debounce_time=1.5
    )
    recognizer.run()