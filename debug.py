import requests

# Mastodon API - no auth required for public accounts
# First get the account ID
resp = requests.get(
    "https://truthsocial.com/api/v1/accounts/lookup",
    params={"acct": "realDonaldTrump"},
    headers={"User-Agent": "Mozilla/5.0"}
)
print("Status:", resp.status_code)
print("Response:", resp.text[:500])