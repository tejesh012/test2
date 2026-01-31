# AI Vision - Face Detection & Emotion Analysis Platform

A modern AI-powered web application for real-time face detection and emotion analysis. Built with Next.js, Flask, PostgreSQL, and MediaPipe machine learning.

## Features

- **Real-time Face Detection**: Advanced ML-powered face detection using MediaPipe
- **Emotion Analysis**: Detects emotions (happy, sad, neutral, surprised, angry) from facial expressions
- **AI Assistance**: Contextual messages based on detected emotions
- **User Authentication**: Secure sign up/login with JWT tokens
- **Dashboard Analytics**: Track detection history and statistics
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Privacy First**: All processing happens securely on your server

## Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations
- **Lucide React** - Beautiful icons
- **shadcn/ui style** - Reusable UI components

### Backend
- **Flask** - Python web framework
- **PostgreSQL** - Relational database
- **Flask-JWT-Extended** - JWT authentication
- **MediaPipe** - Google's ML solution for face detection
- **OpenCV** - Computer vision library

## Prerequisites

Before running locally, ensure you have:

- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.9 or higher) - [Download](https://python.org/)
- **PostgreSQL** (v14 or higher) - [Download](https://postgresql.org/)
- **pip** - Python package manager (comes with Python)

## Local Development Setup

### Step 1: Clone and Navigate

```bash
cd /home/bunny/majorproj
```

### Step 2: Set Up PostgreSQL Database

1. Start PostgreSQL service:
```bash
# Ubuntu/Debian
sudo systemctl start postgresql

# macOS (Homebrew)
brew services start postgresql
```

2. Create the database:
```bash
# Connect to PostgreSQL
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE ai_demo_db;
CREATE USER postgres WITH PASSWORD 'postgres';
GRANT ALL PRIVILEGES ON DATABASE ai_demo_db TO postgres;
\q
```

Or use the default postgres user if already configured.

### Step 3: Set Up Backend (Flask API)

1. Navigate to backend directory:
```bash
cd backend
```

2. Create Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy example env file (already created)
cp .env.example .env

# Edit .env if needed (default values work for local development)
```

5. Initialize database tables:
```bash
python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all()"
```

6. Run the Flask server:
```bash
python run.py
```

The backend will be running at: **http://localhost:5000**

You can test it by visiting: http://localhost:5000/api/health

### Step 4: Set Up Frontend (Next.js)

Open a new terminal window:

1. Navigate to frontend directory:
```bash
cd /home/bunny/majorproj/frontend
```

2. Install npm dependencies:
```bash
npm install
```

3. The environment file is already configured (.env.local):
```
NEXT_PUBLIC_API_URL=http://localhost:5000/api
```

4. Run the development server:
```bash
npm run dev
```

The frontend will be running at: **http://localhost:3000**

## Running Both Servers

You'll need two terminal windows:

**Terminal 1 (Backend):**
```bash
cd /home/bunny/majorproj/backend
source venv/bin/activate
python run.py
```

**Terminal 2 (Frontend):**
```bash
cd /home/bunny/majorproj/frontend
npm run dev
```

## Usage Guide

1. **Visit the App**: Open http://localhost:3000 in your browser

2. **Create Account**: Click "Get Started" and register a new account

3. **Access Dashboard**: After login, you'll be redirected to the dashboard

4. **Enable Camera**: Click "Start Camera" to enable webcam

5. **See Detection**: The app will automatically detect your face and analyze emotions

6. **View History**: Check your detection history and statistics on the dashboard

## Project Structure

```
majorproj/
├── frontend/                 # Next.js React application
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   │   ├── auth/        # Login & Register pages
│   │   │   ├── dashboard/   # Dashboard page
│   │   │   └── page.tsx     # Landing page
│   │   ├── components/      # Reusable components
│   │   │   ├── ui/          # UI primitives (Button, Card, etc.)
│   │   │   └── blocks/      # Complex components
│   │   ├── contexts/        # React contexts
│   │   ├── services/        # API services
│   │   └── lib/             # Utilities
│   └── package.json
│
├── backend/                  # Flask Python API
│   ├── app/
│   │   ├── routes/          # API endpoints
│   │   │   ├── auth.py      # Authentication routes
│   │   │   ├── user.py      # User profile routes
│   │   │   └── face_detection.py  # ML detection routes
│   │   ├── models/          # Database models
│   │   ├── ml/              # Machine learning services
│   │   │   └── face_detection.py  # MediaPipe integration
│   │   └── config.py        # Configuration
│   ├── requirements.txt
│   └── run.py               # Entry point
│
└── README.md
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/logout` - Logout user
- `GET /api/auth/me` - Get current user

### User
- `GET /api/user/dashboard` - Get dashboard data
- `PUT /api/user/profile` - Update profile
- `GET /api/user/detection-history` - Get detection history

### Face Detection
- `POST /api/detection/analyze` - Analyze frame (authenticated)
- `POST /api/detection/analyze-guest` - Analyze frame (guest)
- `GET /api/detection/stats` - Get detection statistics

## Environment Variables

### Backend (.env)
```
FLASK_ENV=development
FLASK_APP=run.py
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_demo_db
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:5000/api
```

## Troubleshooting

### PostgreSQL Connection Error
- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Check database exists: `psql -U postgres -l`
- Verify credentials in `.env` file

### Camera Not Working
- Ensure browser has camera permissions
- Try using HTTPS in production (camera requires secure context)
- Check if another app is using the camera

### Module Not Found (Python)
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### npm Errors
- Clear cache: `npm cache clean --force`
- Delete node_modules: `rm -rf node_modules && npm install`

## Building for Production

### Frontend
```bash
cd frontend
npm run build
npm start
```

### Backend
```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## Security Notes

- Change `SECRET_KEY` and `JWT_SECRET_KEY` in production
- Use HTTPS in production
- Set proper CORS origins for production domains
- Never commit `.env` files with real credentials

## License

MIT License - feel free to use this project for learning and development.

## Credits

- [MediaPipe](https://mediapipe.dev/) - Face detection ML
- [Next.js](https://nextjs.org/) - React framework
- [Flask](https://flask.palletsprojects.com/) - Python web framework
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Framer Motion](https://www.framer.com/motion/) - Animation library
- [Lucide Icons](https://lucide.dev/) - Icon library
