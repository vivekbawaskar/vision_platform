# Deploying: GitHub + Vercel (and where the app itself can run)

## The important part first

**Vercel cannot run this Streamlit app.** Vercel runs short-lived serverless
functions and static sites. This app needs a long-running Python process, a
persistent websocket, PyTorch (far above Vercel's function size limit) and, for
the webcam option, access to a camera. So the split is:

| What | Where |
|---|---|
| Source code, history, CI | **GitHub** |
| Portfolio landing page (`site/`) | **Vercel** |
| The running Streamlit app | Hugging Face Spaces (Docker), Render, Railway, or Streamlit Community Cloud |
| Database for a hosted app | `DB_BACKEND=sqlite` (simplest) or a cloud MySQL |

Also remember: on a hosted server, **"Webcam" reads the server's camera, not
yours.** Use *Upload Video* (or RTSP) for hosted demos. The Dockerfile already
defaults to that.

---

## 1. Push to GitHub

```bash
cd vision_platform

# 1. Make sure secrets are ignored BEFORE the first commit
git init
git check-ignore -v .env        # must print a rule; if it prints nothing, stop and fix .gitignore

# 2. Commit
git add .
git status                      # confirm .env and *.pt are NOT listed
git commit -m "Real-time object detection platform"
git branch -M main
```

Create an empty repository on github.com (no README/licence, you already have
one), then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

(With the GitHub CLI instead: `gh repo create <repo-name> --public --source=. --push`.)

Notes:
- If you ever committed `.env` by accident, **change the password** - deleting
  the file later does not remove it from history.
- Files over 100 MB are rejected by GitHub. Keep `sample.mp4` small.
- The included workflow (`.github/workflows/ci.yml`) runs the tests on every
  push; check the **Actions** tab for a green tick.

---

## 2. Landing page on Vercel

1. Open `site/index.html` and replace the three `YOUR_..._URL` links.
2. Commit and push.
3. On vercel.com: **Add New → Project → Import** your GitHub repo.
4. Set **Root Directory** to `site`, **Framework Preset** to *Other*
   (no build command, no output directory). Click **Deploy**.

Or from the terminal: `cd site && npx vercel` (then `npx vercel --prod`).

Every push to `main` redeploys automatically.

---

## 3. Hosting the real app

### Option A - Hugging Face Spaces (Docker) - most reliable free option
1. huggingface.co → **New Space** → SDK **Docker** → Blank.
2. Push this project to the Space's git repo.
3. The Space reads its settings from the top of `README.md`. Add this block as
   the very first lines **in the Space's copy only** (it would look odd on GitHub):
   ```yaml
   ---
   title: Vision Platform
   sdk: docker
   app_port: 7860
   ---
   ```
4. The first build is slow (PyTorch + model download). Afterwards open the
   Space URL, choose **Upload Video**, press **Start**.

The Dockerfile uses `DB_BACKEND=auto` and a SQLite file in `/tmp`, so logging
works without any database server (it resets when the Space restarts).

### Option B - Render / Railway
Create a *Web Service* from your GitHub repo using the **Dockerfile**. They
inject `PORT` automatically. Add a persistent disk or a cloud MySQL if you want
logs to survive restarts.

### Option C - Streamlit Community Cloud
1. share.streamlit.io → **New app** → pick the repo, main file `main.py`.
2. `packages.txt` (included) installs the system libraries OpenCV needs.
3. In **Advanced settings → Secrets** add (top-level keys become env vars):
   ```toml
   DB_BACKEND = "sqlite"
   DEFAULT_SOURCE = "Upload Video"
   ```
This can work but PyTorch is heavy; if the build fails or the app is killed for
memory, use Option A.

### Using a cloud MySQL instead of SQLite
Create a MySQL instance at a provider (for example Aiven, TiDB Cloud or Railway;
check their current free-tier limits), then set these as secrets / environment
variables on your host:

```
DB_BACKEND=mysql
DB_HOST=...  DB_PORT=...  DB_USER=...  DB_PASSWORD=...  DB_NAME=...
DB_SSL_CA=/path/to/ca.pem      # if the provider requires TLS with a CA file
```
Use `DB_BACKEND=auto` if you want a SQLite fallback when MySQL is unreachable.
