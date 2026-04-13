Sentimate Trader
================

A small Python project that listens to sentiment signals and executes trades. This repository contains the core pieces: a classifier, feed listener, market poller, and an order executor. Logs and state files are kept locally and should not be committed to source control.

Quick overview
--------------
- classifier.py — sentiment classifier and signal extraction
- feed_listener.py — ingests live feeds and forwards messages
- market_poller.py — polls market data
- order_executor.py — places orders with broker/exchange
- debug.py, test_pipeline.py, test_context.py — dev and test utilities
- auth_state.json, seen_ids.json, signal_log.jsonl, trade_log.jsonl — local state and logs (ignore in VCS)

Prerequisites
-------------
- Python 3.10+ (project currently runs on Python 3.12 in developer environment)
- pip for installing any dependencies

Setup (local)
-------------
1. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies (add a requirements.txt later if needed):

   pip install -r requirements.txt

3. Ensure sensitive files are ignored by git (see .gitignore).

Initialize git and push to GitHub
--------------------------------
Option A — using the GitHub website
1. Create a new repository on GitHub (choose public or private).
2. Follow the instructions GitHub shows to push an existing repository, for example:

   git init
   git add .
   git commit -m "chore: initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main

Option B — using the GitHub CLI (gh)

   gh repo create <your-username>/<repo-name> --public --source=. --remote=origin --push

Security note
-------------
- Do NOT commit secrets or state files. Add the following to .gitignore: auth_state.json, signal_log.jsonl, trade_log.jsonl, seen_ids.json, and other runtime logs.
- Consider using environment variables or a secrets manager to store API keys.

Suggested repo layout improvements
---------------------------------
- Add a `requirements.txt` or `pyproject.toml` if using Poetry/Flit
- Add a `tests/` directory and CI (GitHub Actions) to run tests automatically
- Add a `LICENSE` file (MIT, Apache-2.0, etc.)

Contributing
------------
- Use feature branches with descriptive names (see naming conventions in CONTRIBUTING.md)
- Open pull requests and include short descriptions and test results

License
-------
Add a `LICENSE` file at the project root. Common choices: MIT, Apache-2.0.

Contact
-------
If you want, tell me how you want the repository named, whether it should be public or private, and which license you prefer. I can create or update files accordingly and help construct the initial commit and remote commands.
