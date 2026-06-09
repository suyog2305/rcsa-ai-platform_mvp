# 🚀 Deployment Guide

Step-by-step instructions to get your MVP live on the public internet. Estimated time: **30-45 minutes**.

## Prerequisites Checklist

- [ ] GitHub account (free)
- [ ] OpenAI API account with $5+ credit ([platform.openai.com](https://platform.openai.com))
- [ ] Railway account (free, no credit card needed) — [railway.app](https://railway.app)
- [ ] Vercel account (free) — [vercel.com](https://vercel.com)

---

## Step 1 — Push Code to GitHub (5 minutes)

```bash
# In the rcsa-ai-platform folder
git init
git add .
git commit -m "Initial commit — RCSA AI Platform MVP"

# Create new repo on github.com (public) called: rcsa-ai-platform
git remote add origin https://github.com/YOUR_USERNAME/rcsa-ai-platform.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Deploy Backend to Railway (15 minutes)

### 2.1 Connect Railway to GitHub
1. Go to [railway.app](https://railway.app) → Sign in with GitHub
2. Click **+ New Project** → **Deploy from GitHub repo**
3. Authorise Railway to access your `rcsa-ai-platform` repository
4. Select the repo

### 2.2 Configure the build
1. Railway will detect Python — but you need to point it at the backend folder
2. Click **Settings** → **Root Directory** → enter `/backend`
3. Click **Settings** → **Start Command** → enter:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

### 2.3 Add environment variable
1. Click **Variables** tab
2. Click **+ New Variable**
3. Name: `OPENAI_API_KEY` · Value: `sk-your-key-here`
4. Click **Add**

### 2.4 Generate public URL
1. Click **Settings** → **Networking** → **Generate Domain**
2. Railway gives you a URL like: `rcsa-ai-platform-production.up.railway.app`
3. **Copy this URL** — you'll need it for the frontend

### 2.5 Test the backend
Open in browser:
```
https://YOUR-RAILWAY-URL.up.railway.app/knowledge_base
```

You should see JSON listing your documents. If you see this, the backend is live and the ChromaDB ingestion worked.

---

## Step 3 — Deploy Frontend to Vercel (10 minutes)

### 3.1 Import repo
1. Go to [vercel.com](https://vercel.com) → Sign in with GitHub
2. Click **Add New** → **Project**
3. Import your `rcsa-ai-platform` repository

### 3.2 Configure deployment
1. **Framework Preset:** Other
2. **Root Directory:** click **Edit** → enter `frontend`
3. **Build Command:** leave empty
4. **Output Directory:** leave empty (Vercel will serve files from the root)
5. Click **Deploy**

### 3.3 Get your live URL
Vercel deploys in 30 seconds and gives you a URL like:
```
https://rcsa-ai-platform.vercel.app
```

### 3.4 Test the live site
1. Open your Vercel URL
2. Enter your Railway backend URL in the "Backend API URL" field
3. Click **Connect**
4. The Knowledge Base panel should populate
5. The agent should send its opening message

---

## Step 4 — Configure Custom Domain (Optional, 5 minutes)

For a more professional look:

### Option A — Vercel's free subdomain
You already have `rcsa-ai-platform.vercel.app`. Free and good enough.

### Option B — Custom domain
1. Buy a domain on [Namecheap](https://namecheap.com) or [Porkbun](https://porkbun.com) (~$12/year)
2. In Vercel project → **Settings** → **Domains** → **Add**
3. Enter your domain → follow DNS setup instructions

---

## Step 5 — Lock Down CORS (Important Before Sharing)

Currently the backend accepts requests from any origin. Before sharing the URL publicly:

1. Open `backend/main.py`
2. Find the CORS middleware section
3. Replace `allow_origins=["*"]` with:
   ```python
   allow_origins=["https://your-vercel-url.vercel.app"]
   ```
4. Commit and push — Railway auto-redeploys

---

## Step 6 — Monitor Your Spending

### OpenAI usage
- Go to [platform.openai.com/usage](https://platform.openai.com/usage)
- Set a monthly spending limit ($20 is plenty)
- Each demo conversation costs ~$0.05-0.15

### Railway usage
- Free tier includes $5/month execution time
- This project uses ~$2/month at moderate load

### Total monthly cost
- **Light demo usage:** ~$3-5/month
- **Heavy showcase usage:** ~$10-15/month

---

## 🐛 Troubleshooting

### Backend won't start
- Check Railway logs (click **Deployments** → click the deployment → **Logs**)
- Most common issue: `OPENAI_API_KEY` not set or invalid

### Frontend can't connect
- Check the backend URL — must include `https://`
- Open browser console (F12) to see the actual error
- Common issue: CORS — make sure you've added your Vercel URL to allowed origins

### Empty knowledge base
- Make sure your `.docx` files are in the `/control_docs` folder at the repo root
- Check that they were pushed to GitHub (don't `.gitignore` them)
- Restart the Railway service to re-trigger ingestion

### "Rate limit exceeded" on OpenAI
- You've hit your spending limit — top up at platform.openai.com/billing
- Or switch to GPT-4o-mini in the code (10x cheaper, slightly less capable)

---

## 📣 Sharing Your MVP

### LinkedIn launch post template

```
🚀 Just shipped an AI-powered RCSA platform — real RAG, real banking documents, real deployment.

The problem: banks spend 87 days per RCSA cycle. By the time the Board sees the assessment, it's 60-90 days stale. In 2024 alone, $4.3B in regulatory penalties were paid for control failures.

The solution: a conversational AI agent that conducts adaptive risk interviews with control owners, retrieves relevant control documentation semantically, and generates structured RCSA output with regulatory citations.

The architecture:
→ GPT-4o for reasoning
→ ChromaDB for vector retrieval
→ FastAPI backend on Railway
→ Vanilla JS frontend on Vercel
→ Real banking control documents (synthetic, no real data)

Watch the demo (90 sec): [Loom URL]
Try it live: [Vercel URL]
Source code: [GitHub URL]

Open to chatting with anyone working on AI in financial services risk management.

#AI #RiskManagement #Banking #RCSA #GenAI #LLM #FinTech
```

---

You're done. You now have a public, working, hireable-worthy AI build on the internet.
