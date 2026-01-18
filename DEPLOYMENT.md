# Deployment Guide

This guide will help you deploy your Stock Dashboard to Streamlit Cloud (free and easy).

## Option 1: Streamlit Cloud (Recommended - Free & Easy)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps:

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**:
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy your app**:
   - Click "New app"
   - Select your repository: `don-herb-stocks`
   - Select branch: `main`
   - Main file path: `dashboard.py`
   - Click "Deploy!"

4. **Your app will be live** at: `https://your-app-name.streamlit.app`

5. **Share with client**: Simply send them the URL!

---

## Option 2: Railway (Alternative - Free Tier Available)

### Steps:

1. **Install Railway CLI** (optional, or use web interface):
   ```bash
   npm i -g @railway/cli
   ```

2. **Go to Railway**:
   - Visit: https://railway.app/
   - Sign up with GitHub

3. **Create new project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

4. **Configure deployment**:
   - Add a service
   - Select "Python" template
   - Set start command: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`
   - Railway will auto-detect requirements.txt

5. **Deploy**: Railway will automatically deploy your app

---

## Option 3: Render (Alternative - Free Tier Available)

### Steps:

1. **Go to Render**:
   - Visit: https://render.com/
   - Sign up with GitHub

2. **Create new Web Service**:
   - Connect your GitHub repository
   - Select "Web Service"
   - Choose your repo

3. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`
   - Environment: Python 3

4. **Deploy**: Click "Create Web Service"

---

## Quick Deploy Commands

If you want to commit and push everything now:

```bash
# Add all files
git add .

# Commit changes
git commit -m "Add deployment configuration"

# Push to GitHub
git push origin main
```

Then follow the Streamlit Cloud steps above!

---

## Notes:

- **Streamlit Cloud** is the easiest option and specifically designed for Streamlit apps
- All options are free for basic usage
- Your app will automatically update when you push to GitHub
- No credit card required for any of these options
