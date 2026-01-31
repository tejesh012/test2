# Sentio - Emotional AI Assistant

Sentio is an emotional AI platform that combines real-time face detection, emotion analysis, and an intelligent chatbot that understands how you feel.

## Features

- **Real-time Face Detection**: Advanced ML-powered face detection using OpenCV Haar Cascades
- **Emotion Analysis**: Detect emotions through facial expressions
- **Emotional AI Chatbot**: ChatGPT-style chatbot that responds based on your detected mood
- **Privacy First**: All processing happens securely with your data protected
- **Interactive 3D Robot**: Spline-powered 3D robot companion on the landing page
- **Beautiful UI**: Modern dark theme with sparkles, animations, and glass morphism effects

## Tech Stack

### Frontend
- **Next.js 15** with App Router
- **React 19** with TypeScript
- **Tailwind CSS 4** for styling
- **Framer Motion** for animations
- **shadcn/ui** inspired components
- **tsparticles** for sparkle effects
- **Spline** for 3D graphics

### Backend
- **Flask** with Python
- **PostgreSQL** database
- **Flask-SQLAlchemy** for ORM
- **Flask-JWT-Extended** for authentication
- **OpenCV** for face detection
- **Flask-CORS** for cross-origin requests

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- PostgreSQL 14+

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a PostgreSQL database:
   ```bash
   createdb sentio_db
   ```

5. Create a `.env` file in the backend directory:
   ```env
   DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/sentio_db
   JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
   FLASK_ENV=development
   FLASK_DEBUG=1
   ```

6. Run database migrations:
   ```bash
   flask db upgrade
   ```

7. Start the backend server:
   ```bash
   flask run --port 5000
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env.local` file:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:5000/api
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
majorproj/
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # App router pages
│   │   │   ├── auth/        # Login/Register pages
│   │   │   ├── dashboard/   # User dashboard
│   │   │   └── page.tsx     # Landing page
│   │   ├── components/      # React components
│   │   │   ├── ui/          # UI primitives (Button, Card, etc.)
│   │   │   └── blocks/      # Complex components
│   │   ├── contexts/        # React contexts (Auth)
│   │   ├── services/        # API services
│   │   └── lib/             # Utility functions
│   └── public/              # Static assets
│
└── backend/                  # Flask backend
    └── app/
        ├── routes/          # API routes
        ├── models/          # Database models
        ├── ml/              # Machine learning (face detection)
        └── config.py        # Configuration
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login and get JWT tokens
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - Logout user

### User
- `GET /api/user/me` - Get current user profile
- `PUT /api/user/me` - Update user profile
- `GET /api/user/dashboard` - Get dashboard data

### Detection
- `POST /api/detection/analyze` - Analyze image for faces and emotions
- `GET /api/detection/history` - Get detection history

## Environment Variables

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:5000/api
```

### Backend (.env)
```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/sentio_db
JWT_SECRET_KEY=your-secret-key
FLASK_ENV=development
FLASK_DEBUG=1
```

## License

MIT License
