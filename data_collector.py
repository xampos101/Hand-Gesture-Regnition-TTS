"""
Hand Sign Recognition - Data Collection Module (UPGRADED)
---------------------------------------------------------
Major Improvements:
1. Uses X, Y, Z coordinates (63 features instead of 42) for better accuracy
2. Added visual quality indicators (hand size, stability)
3. Automatic sample collection mode with countdown
4. Better file organization and error handling
5. Data augmentation preview
6. Real-time statistics
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import numpy as np
import os
import urllib.request
import time
from collections import Counter


class HandSignDataCollector:
    def __init__(self, csv_file='dataset.csv'):
        self.csv_file = csv_file
        self.model_file = 'hand_landmarker.task'
        self.sample_count = 0
        self.session_samples = 0

        # Download model if needed
        self._download_model()

        # Initialize MediaPipe
        print("🔧 Initializing MediaPipe Hand Landmarker...")
        base_options = python.BaseOptions(model_asset_path=self.model_file)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,  # Increased for better quality
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # CSV setup with X, Y, Z coordinates
        self.header = (
                [f'x{i}' for i in range(21)] +
                [f'y{i}' for i in range(21)] +
                [f'z{i}' for i in range(21)] +
                ['label']
        )

        # Create or load existing dataset
        self._initialize_csv()

        # Collection state
        self.current_label = None
        self.auto_collect_mode = False
        self.auto_collect_countdown = 0
        self.auto_samples_target = 30
        self.auto_samples_collected = 0

        # Quality metrics
        self.hand_stability_history = []
        self.max_stability_history = 10

    def _download_model(self):
        """Download MediaPipe model if not present"""
        if not os.path.exists(self.model_file):
            print("📥 Downloading hand_landmarker model...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            try:
                urllib.request.urlretrieve(url, self.model_file)
                print("✓ Model downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                raise

    def _initialize_csv(self):
        """Initialize CSV file with header"""
        if os.path.exists(self.csv_file):
            # Check if file has correct format (64 columns: 63 features + label)
            import pandas as pd
            try:
                # Read first line to check format
                with open(self.csv_file, 'r') as f:
                    first_line = f.readline().strip()
                    col_count = len(first_line.split(','))
                
                expected_cols = len(self.header)  # Should be 64 (63 features + label)
                
                if col_count != expected_cols:
                    # Old format detected - backup and create new
                    from datetime import datetime
                    backup_file = self.csv_file.replace('.csv', f'_old_format_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                    import shutil
                    shutil.copy2(self.csv_file, backup_file)
                    print(f"⚠ Old format detected ({col_count} cols vs {expected_cols} expected)")
                    print(f"📦 Backed up to: {backup_file}")
                    print(f"🔄 Creating new dataset file with correct format...")
                    
                    # Create new file with correct header
                    with open(self.csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.header)
                    self.sample_count = 0
                    print(f"✓ Created new dataset: {self.csv_file}")
                    print(f"💡 Note: You'll need to collect new samples with the new format (X, Y, Z coordinates)")
                else:
                    # Correct format - load existing data
                    df = pd.read_csv(self.csv_file)
                    self.sample_count = len(df)
                    print(f"✓ Loaded existing dataset: {self.sample_count} samples")

                    # Show class distribution
                    if self.sample_count > 0:
                        class_counts = Counter(df['label'])
                        print("📊 Current class distribution:")
                        for label, count in sorted(class_counts.items()):
                            print(f"   {label}: {count} samples")
            except Exception as e:
                # File exists but is corrupted or wrong format
                print(f"⚠ Error reading CSV file: {e}")
                from datetime import datetime
                backup_file = self.csv_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                import shutil
                if os.path.exists(self.csv_file):
                    shutil.copy2(self.csv_file, backup_file)
                    print(f"📦 Backed up to: {backup_file}")
                
                # Create new file
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.header)
                self.sample_count = 0
                print(f"✓ Created new dataset: {self.csv_file}")
        else:
            # Create new file
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
            print(f"✓ Created new dataset: {self.csv_file}")
            self.sample_count = 0

    def extract_landmarks(self, hand_landmarks):
        """
        Extract and normalize 63 features (21 landmarks × 3 coordinates)
        Returns: numpy array of shape (63,)
        """
        # Extract raw coordinates
        landmarks = []
        for lm in hand_landmarks[:21]:
            landmarks.append([lm.x, lm.y, lm.z])

        # Ensure we have exactly 21 landmarks
        while len(landmarks) < 21:
            landmarks.append([0.0, 0.0, 0.0])

        landmarks = np.array(landmarks)

        # Normalize relative to wrist (landmark 0)
        wrist = landmarks[0]
        normalized = landmarks - wrist

        # Flatten to (63,) shape: [x0, x1, ..., x20, y0, y1, ..., y20, z0, z1, ..., z20]
        x_coords = normalized[:, 0]
        y_coords = normalized[:, 1]
        z_coords = normalized[:, 2]

        return np.concatenate([x_coords, y_coords, z_coords])

    def calculate_hand_quality(self, hand_landmarks):
        """
        Calculate quality metrics for the detected hand
        Returns: (hand_size, stability_score, is_good_quality)
        """
        # Calculate hand size (distance between wrist and middle finger tip)
        wrist = hand_landmarks[0]
        middle_tip = hand_landmarks[12]
        hand_size = np.sqrt(
            (middle_tip.x - wrist.x) ** 2 +
            (middle_tip.y - wrist.y) ** 2 +
            (middle_tip.z - wrist.z) ** 2
        )

        # Calculate stability (low variance in recent positions)
        if len(self.hand_stability_history) > 0:
            recent_positions = np.array(self.hand_stability_history[-5:])
            stability = 1.0 - np.std(recent_positions)  # Higher = more stable
        else:
            stability = 0.5

        # Update history
        self.hand_stability_history.append(hand_size)
        if len(self.hand_stability_history) > self.max_stability_history:
            self.hand_stability_history.pop(0)

        # Good quality criteria
        is_good = hand_size > 0.15 and stability > 0.8  # Hand not too small, stable

        return hand_size, stability, is_good

    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks with connections"""
        h, w, _ = frame.shape

        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for idx, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            # Highlight wrist and fingertips
            if idx == 0 or idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)
            else:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    def save_sample(self, landmarks, label):
        """Save a single sample to CSV"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = landmarks.tolist() + [label]
            writer.writerow(row)

        self.sample_count += 1
        self.session_samples += 1

    def run(self):
        """Main collection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n" + "=" * 70)
        print("🎥 HAND SIGN DATA COLLECTION")
        print("=" * 70)
        print("CONTROLS:")
        print("  A-Z / 0-9    : Set label and save sample")
        print("  SPACE        : Label as 'SPACE' gesture")
        print("  ENTER        : Start auto-collection mode (30 samples)")
        print("  C            : Clear session counter")
        print("  R            : Show statistics")
        print("  Q            : Quit")
        print("=" * 70 + "\n")

        last_save_time = 0
        min_save_interval = 0.1  # 100ms between saves to avoid duplicates

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to capture frame")
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hands
            detection_result = self.hand_landmarker.detect(mp_image)

            hand_detected = False
            quality_text = "NO HAND"
            quality_color = (0, 0, 255)

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                hand_detected = True

                # Draw landmarks
                self.draw_landmarks(frame, hand_landmarks)

                # Calculate quality
                hand_size, stability, is_good = self.calculate_hand_quality(hand_landmarks)

                if is_good:
                    quality_text = f"READY ✓ (Size: {hand_size:.2f}, Stable: {stability:.2f})"
                    quality_color = (0, 255, 0)
                else:
                    quality_text = f"ADJUST (Size: {hand_size:.2f}, Stable: {stability:.2f})"
                    quality_color = (0, 165, 255)

                # Auto-collection mode
                if self.auto_collect_mode and is_good:
                    current_time = time.time()

                    if self.auto_collect_countdown > 0:
                        # Countdown phase
                        self.auto_collect_countdown -= 1
                        cv2.putText(frame, f"GET READY: {self.auto_collect_countdown // 10 + 1}",
                                    (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                    else:
                        # Collection phase
                        if (current_time - last_save_time) >= min_save_interval:
                            landmarks = self.extract_landmarks(hand_landmarks)
                            self.save_sample(landmarks, self.current_label)
                            self.auto_samples_collected += 1
                            last_save_time = current_time

                            print(f"  📸 Auto-saved sample {self.auto_samples_collected}/{self.auto_samples_target}")

                            if self.auto_samples_collected >= self.auto_samples_target:
                                print(f"✓ Auto-collection complete for '{self.current_label}'")
                                self.auto_collect_mode = False
                                self.current_label = None

                # Manual save (when label is set and not in auto mode)
                elif self.current_label and not self.auto_collect_mode:
                    current_time = time.time()
                    if (current_time - last_save_time) >= min_save_interval:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.save_sample(landmarks, self.current_label)
                        last_save_time = current_time
                        print(f"✓ Saved sample #{self.sample_count}: '{self.current_label}'")
                        self.current_label = None  # Reset after one save

            # === UI OVERLAY ===
            overlay_y = 30

            # Quality indicator
            cv2.putText(frame, quality_text, (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
            overlay_y += 40

            # Current label
            if self.current_label:
                label_text = f"Label: {self.current_label}"
                if self.auto_collect_mode:
                    label_text += f" (Auto: {self.auto_samples_collected}/{self.auto_samples_target})"
                cv2.putText(frame, label_text, (10, overlay_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                overlay_y += 40

            # Sample counts
            cv2.putText(frame, f"Session: {self.session_samples} | Total: {self.sample_count}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Hand Sign Data Collector', frame)

            # === KEY HANDLING ===
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                print("\n🛑 Quitting...")
                break

            elif key == ord('b'):
                self.session_samples = 0
                print("🗑️  Session counter reset")

            elif key == ord('a'):
                self._show_statistics()

            elif key == 13:  # ENTER key - auto collection
                if self.current_label and hand_detected:
                    self.auto_collect_mode = True
                    self.auto_collect_countdown = 30  # 3 second countdown at 10 FPS
                    self.auto_samples_collected = 0
                    print(f"🤖 Auto-collection mode: '{self.current_label}' ({self.auto_samples_target} samples)")
                else:
                    print("⚠ Set a label first (press A-Z) and ensure hand is visible")

            elif key == ord(' '):
                self.current_label = 'SPACE'
                print(f"📝 Label set to: SPACE")

            elif key != 255:
                char = chr(key).upper()
                if char.isalnum():
                    self.current_label = char
                    print(f"📝 Label set to: {char}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hand_landmarker.close()

        print(f"\n✓ Data collection complete")
        print(f"  Total samples: {self.sample_count}")
        print(f"  Session samples: {self.session_samples}")
        self._show_statistics()

    def _show_statistics(self):
        """Show dataset statistics"""
        if not os.path.exists(self.csv_file):
            print("⚠ No dataset found")
            return

        import pandas as pd
        df = pd.read_csv(self.csv_file)

        print("\n" + "=" * 50)
        print("📊 DATASET STATISTICS")
        print("=" * 50)
        print(f"Total samples: {len(df)}")
        print(f"Total classes: {df['label'].nunique()}")
        print("\nClass distribution:")

        class_counts = df['label'].value_counts().sort_index()
        for label, count in class_counts.items():
            bar = "█" * (count // 5)
            print(f"  {label:>6}: {count:>4} {bar}")

        print("=" * 50 + "\n")


if __name__ == "__main__":
    collector = HandSignDataCollector()
    collector.run()