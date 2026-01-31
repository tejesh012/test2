# Deployment Guide

This project is configured for deployment using a separated frontend (Vercel) and backend (Render/Railway) strategy.

## Prerequisites

- [GitHub](https://github.com/) account
- [Render](https://render.com/) or [Railway](https://railway.app/) account (for Backend & Database)
- [Vercel](https://vercel.com/) account (for Frontend)
- Git installed locally

## Step 1: Push to GitHub

If you haven't already, push your code to a GitHub repository.

```bash
git add .
git commit -m "Prepare for deployment"
git push origin master
```

## Step 2: Deploy Backend (Render)

1.  **Create a Database**:
    *   Log in to Render dashboard.
    *   Click "New +" -> "PostgreSQL".
    *   Name it (e.g., `ai-demo-db`).
    *   Copy the `Internal Database URL` (for internal networking) or just wait for the web service step where you can link them.

2.  **Create Web Service**:
    *   Click "New +" -> "Web Service".
    *   Connect your GitHub repository.
    *   **Root Directory**: `backend`
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn run:app`
    *   **Environment Variables**:
        *   `SECRET_KEY`: (Generate a random string)
        *   `JWT_SECRET_KEY`: (Generate a random string)
        *   `PYTHON_VERSION`: `3.11.0` (Recommended)
        *   `DATABASE_URL`: (Paste the connection string from your PostgreSQL database. If you created them in the same Render account/project, you might be able to link them directly).

3.  **Deploy**:
    *   Click "Create Web Service".
    *   Wait for the deployment to finish.
    *   **Copy the backend URL** (e.g., `https://your-app.onrender.com`).

## Step 3: Deploy Frontend (Vercel)

1.  **Import Project**:
    *   Log in to Vercel.
    *   Click "Add New..." -> "Project".
    *   Import your GitHub repository.

2.  **Configure Project**:
    *   **Framework Preset**: Next.js (should be auto-detected).
    *   **Root Directory**: Edit this and select `frontend`.
    *   **Environment Variables**:
        *   `NEXT_PUBLIC_API_URL`: Paste your Render Backend URL (e.g., `https://your-app.onrender.com`). **Important**: Do not add a trailing slash if your code appends `/api`, but based on current config, the backend URL should probably look like `https://your-app.onrender.com/api` or just the base depending on how `api.ts` uses it.
        *   *Check `frontend/src/services/api.ts`*: It uses `process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api'`. So set `NEXT_PUBLIC_API_URL` to `https://your-app.onrender.com/api`.

3.  **Deploy**:
    *   Click "Deploy".

## Step 4: Final Configuration

1.  **Update Backend CORS**:
    *   Go back to your Render Web Service dashboard.
    *   Add/Update Environment Variable:
        *   `FRONTEND_URL`: Paste your Vercel URL (e.g., `https://your-project.vercel.app`).
    *   This ensures the backend allows requests from your deployed frontend.

2.  **Redeploy Backend**:
    *   Render typically restarts automatically when env vars change. If not, trigger a manual deploy.

## Troubleshooting

-   **Database Connection**: Ensure `DATABASE_URL` starts with `postgresql://`. The code handles the `postgres://` fix automatically.
-   **CORS Errors**: Check the browser console. If you see CORS errors, verify `FRONTEND_URL` in backend matches your Vercel domain exactly (no trailing slash usually).
-   **Build Failures**: Check logs in Render/Vercel.
