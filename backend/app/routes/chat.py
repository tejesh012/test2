from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
import re

chat_bp = Blueprint('chat', __name__)

# Knowledge base for Sentio
KNOWLEDGE_BASE = {
    "login": {
        "keywords": ["login", "sign in", "log in", "signin", "access account", "can't login", "how to login"],
        "response": """To login to Sentio:

1. Click the **Login** button in the top right corner or go to /auth/login
2. Enter your email address
3. Enter your password
4. Click **Sign In**

If you don't have an account yet, click **Register** to create one first.

Having trouble? Make sure:
- Your email is correct
- Your password is at least 8 characters
- Caps lock is off"""
    },
    "register": {
        "keywords": ["register", "sign up", "signup", "create account", "new account", "join"],
        "response": """To create a Sentio account:

1. Click **Register** on the homepage or go to /auth/register
2. Fill in your details:
   - First and Last name
   - Username (unique)
   - Email address
   - Password (min 8 characters)
3. Click **Create Account**

You'll be automatically logged in after registration!"""
    },
    "emotion": {
        "keywords": ["emotion", "detect emotion", "feeling", "mood", "how does emotion", "emotion detection", "facial expression"],
        "response": """Sentio uses advanced AI to detect your emotions:

**How it works:**
1. Enable your camera in the Dashboard
2. Sentio analyzes your facial expressions in real-time
3. It detects emotions like: Happy, Sad, Angry, Surprised, Neutral
4. The AI adapts its responses based on how you're feeling

**Privacy:** All processing happens securely. We don't store your camera feed."""
    },
    "camera": {
        "keywords": ["camera", "webcam", "video", "enable camera", "start camera", "camera not working"],
        "response": """To use the camera feature:

1. Go to **Dashboard**
2. Click **Start Camera** button
3. Allow camera permissions when prompted
4. Your face will be detected automatically

**Troubleshooting:**
- Make sure no other app is using your camera
- Check browser permissions (click lock icon in address bar)
- Try refreshing the page
- Use good lighting for better detection"""
    },
    "dashboard": {
        "keywords": ["dashboard", "main page", "control panel", "features", "what can i do"],
        "response": """The Sentio Dashboard gives you access to:

1. **Live Camera Feed** - Real-time face detection
2. **Emotion Analysis** - See your detected emotions
3. **AI Chat** - Talk to Sentio about anything
4. **Detection History** - View past emotion readings
5. **Stats** - Track your total detections

Navigate using the sidebar on the left!"""
    },
    "help": {
        "keywords": ["help", "how to use", "guide", "tutorial", "instructions", "what is sentio"],
        "response": """**Welcome to Sentio - Your Emotional AI Companion!**

Sentio helps you understand and track your emotions using AI.

**Key Features:**
- 🎥 Real-time face detection via webcam
- 😊 Emotion analysis (happy, sad, angry, etc.)
- 💬 AI chatbot that responds to your mood
- 📊 Track your emotional patterns

**Getting Started:**
1. Login or Register for an account
2. Go to Dashboard
3. Enable your camera
4. Start chatting with me!

Ask me anything - I'm here to help!"""
    },
    "about": {
        "keywords": ["about", "what is", "who made", "technology", "how does it work"],
        "response": """**About Sentio**

Sentio is an emotional AI platform that combines:
- **Face Detection** using OpenCV and Haar Cascades
- **Emotion Analysis** through facial expression recognition
- **AI Chatbot** that adapts to your emotional state

**Tech Stack:**
- Frontend: Next.js, React, Tailwind CSS
- Backend: Flask, PostgreSQL
- ML: OpenCV for face detection

Sentio is designed to provide empathetic support and help you understand your emotions better."""
    },
    "privacy": {
        "keywords": ["privacy", "data", "secure", "safe", "store", "information"],
        "response": """**Your Privacy Matters**

Sentio takes your privacy seriously:

✅ Camera feed is processed in real-time, not stored
✅ Only detection results are saved (not images)
✅ Your data is encrypted
✅ We never share your information
✅ You can delete your account anytime

All emotion detection happens through secure API calls."""
    }
}

# Emotion-specific responses
EMOTION_RESPONSES = {
    "happy": [
        "You seem to be in a great mood! That's wonderful to see. How can I help you today?",
        "Your happiness is contagious! What exciting things are you working on?",
        "I love seeing that positive energy! What would you like to know?"
    ],
    "sad": [
        "I notice you might be feeling down. I'm here to help however I can. What's on your mind?",
        "It's okay to have difficult moments. Would you like to talk about it, or can I help you with something?",
        "I'm here for you. Sometimes just chatting can help. What do you need?"
    ],
    "angry": [
        "I sense some frustration. Let me help you with whatever you need.",
        "Take a deep breath. I'm here to assist. What can I do for you?",
        "I understand things can be frustrating. How can I help resolve this?"
    ],
    "surprised": [
        "Something caught your attention! What would you like to know more about?",
        "Curious about something? I'm happy to explain!",
        "I see you're intrigued! How can I help?"
    ],
    "neutral": [
        "Hello! I'm Sentio, your AI assistant. How can I help you today?",
        "Hi there! What would you like to know?",
        "Welcome! I'm here to answer your questions about Sentio."
    ]
}

def find_best_response(message: str, emotion: str = "neutral") -> str:
    """Find the best response based on message content and emotion."""
    message_lower = message.lower().strip()

    # Check knowledge base first
    best_match = None
    best_score = 0

    for topic, data in KNOWLEDGE_BASE.items():
        for keyword in data["keywords"]:
            if keyword in message_lower:
                # Score based on keyword specificity
                score = len(keyword)
                if score > best_score:
                    best_score = score
                    best_match = data["response"]

    if best_match:
        return best_match

    # Check for greetings
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(greet in message_lower for greet in greetings):
        return f"""Hello! Welcome to Sentio! 👋

I'm your emotional AI assistant. I can help you with:
- **Login/Register** - Account access
- **Camera & Detection** - Face and emotion detection
- **Dashboard** - Using Sentio's features
- **Privacy** - How we protect your data

What would you like to know?"""

    # Check for thanks
    if any(word in message_lower for word in ["thank", "thanks", "thx"]):
        return "You're welcome! Is there anything else I can help you with?"

    # Check for goodbye
    if any(word in message_lower for word in ["bye", "goodbye", "see you", "later"]):
        return "Goodbye! Remember, I'm always here when you need help. Take care! 👋"

    # Check for questions about capabilities
    if "can you" in message_lower or "do you" in message_lower:
        return """I can help you with:

• **Account Help** - Login, register, password issues
• **Feature Guides** - Camera, emotion detection, dashboard
• **Technical Support** - Troubleshooting problems
• **General Questions** - About Sentio and how it works

Just ask me anything specific!"""

    # Default response based on emotion
    import random
    emotion_responses = EMOTION_RESPONSES.get(emotion, EMOTION_RESPONSES["neutral"])
    return random.choice(emotion_responses) + "\n\nTry asking about: login, camera, emotions, or dashboard features."


@chat_bp.route('/message', methods=['POST'])
def send_message():
    """Process a chat message and return AI response."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        emotion = data.get('emotion', 'neutral')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Generate response
        response = find_best_response(message, emotion)

        return jsonify({
            'success': True,
            'response': response,
            'emotion_detected': emotion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested questions based on context."""
    suggestions = [
        "How do I login?",
        "How does emotion detection work?",
        "How to use the camera?",
        "What is Sentio?",
        "What features are available?",
        "Is my data private?"
    ]
    return jsonify({'suggestions': suggestions})
