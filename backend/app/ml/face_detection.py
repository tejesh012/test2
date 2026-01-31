"""
Face Detection and Analysis Service using OpenCV
Simple ML model for face detection and basic emotion inference
"""

import cv2
import numpy as np
import base64
from typing import Dict, List, Optional, Tuple


class FaceDetectionService:
    """Service for face detection and analysis using OpenCV"""

    def __init__(self):
        # Load OpenCV's pre-trained face detection model (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load eye cascade for additional analysis
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # Load smile cascade for emotion detection
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )

        # Emotion mapping
        self.emotions = ['neutral', 'happy', 'sad', 'surprised', 'angry']

    def decode_base64_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Decode base64 string to numpy array image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            image_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in an image and return bounding boxes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_list = []
        for (x, y, w, h) in faces:
            # Calculate confidence based on face size relative to image
            img_area = image.shape[0] * image.shape[1]
            face_area = w * h
            confidence = min(0.95, (face_area / img_area) * 10 + 0.5)

            face_list.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'confidence': float(confidence)
            })

        return face_list

    def analyze_face(self, image: np.ndarray, face: Dict) -> Dict:
        """Analyze a detected face for emotions"""
        x, y, w, h = face['x'], face['y'], face['width'], face['height']

        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 3)
        eyes_detected = len(eyes)

        # Detect smile (in lower half of face)
        lower_face = gray_roi[h//2:, :]
        smiles = self.smile_cascade.detectMultiScale(
            lower_face,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        smile_detected = len(smiles) > 0

        # Infer emotion based on detections
        emotion, confidence = self._infer_emotion(eyes_detected, smile_detected, face)

        return {
            'emotion': emotion,
            'emotion_confidence': confidence,
            'eyes_detected': eyes_detected,
            'smile_detected': smile_detected,
            'landmark_count': eyes_detected * 2 + (1 if smile_detected else 0)
        }

    def _infer_emotion(self, eyes_detected: int, smile_detected: bool, face: Dict) -> Tuple[str, float]:
        """
        Simple emotion inference based on facial features.
        """
        # Calculate face aspect ratio
        aspect_ratio = face['width'] / face['height'] if face['height'] > 0 else 1

        if smile_detected and eyes_detected >= 2:
            return 'happy', 0.85
        elif smile_detected:
            return 'happy', 0.7
        elif eyes_detected == 0:
            # Eyes closed or looking down could indicate sadness
            return 'sad', 0.5
        elif aspect_ratio > 1.2:
            # Wide face could indicate surprise (raised eyebrows)
            return 'surprised', 0.6
        elif eyes_detected >= 2:
            return 'neutral', 0.75
        else:
            return 'neutral', 0.5

    def process_frame(self, base64_image: str) -> Dict:
        """Process a single frame for face detection and analysis"""
        image = self.decode_base64_image(base64_image)

        if image is None:
            return {
                'success': False,
                'error': 'Failed to decode image'
            }

        # Detect faces
        faces = self.detect_faces(image)

        # Analyze each face
        analysis = []
        for face in faces:
            face_analysis = self.analyze_face(image, face)
            analysis.append(face_analysis)

        return {
            'success': True,
            'faces_detected': len(faces),
            'faces': faces,
            'analysis': analysis,
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            }
        }

    def get_assistance_message(self, analysis: List[Dict]) -> str:
        """Generate assistance message based on detected emotions"""
        if not analysis:
            return "No face detected. Please ensure your face is visible in the camera."

        primary_emotion = analysis[0].get('emotion', 'neutral')

        messages = {
            'happy': "You seem to be in a good mood! How can I assist you today?",
            'sad': "I notice you might be feeling down. Is there anything I can help with?",
            'surprised': "Something caught your attention! What would you like to know?",
            'angry': "I sense some frustration. Let me know how I can help resolve any issues.",
            'neutral': "Hello! I'm here to help. What would you like to do today?"
        }

        return messages.get(primary_emotion, messages['neutral'])


# Singleton instance
face_detection_service = FaceDetectionService()
