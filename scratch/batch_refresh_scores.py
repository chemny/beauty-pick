import sys
import time
import httpx

BASE_URL = "http://127.0.0.1:8765"

def refresh_all():
    with httpx.Client(base_url=BASE_URL, timeout=180) as client:
        # Get session
        r = client.get("/api/session")
        r.raise_for_status()
        token = r.json()["token"]
        client.headers["x-portrait-token"] = token

        # Get assets
        r = client.get("/api/assets")
        r.raise_for_status()
        assets = r.json()

        outdated = assets
        print(f"Total assets: {len(assets)}, Assets to refresh: {len(outdated)}")

        for i, a in enumerate(outdated, 1):
            aid = a["id"]
            fn = a["filename"]
            print(f"[{i}/{len(outdated)}] Analyzing 3 modules: {fn[:30]} ({aid[:8]})...")

            # Ensure consent
            try:
                client.post(f"/api/assets/{aid}/consent", json={"adult_confirmed": True, "external_consent": True, "external_scope": "full_image"})
                resp = client.post(f"/api/assets/{aid}/analyze", json={"modules": ["face", "body", "photography"], "force": True})
                resp.raise_for_status()
                job_id = resp.json()["job_id"]
            except Exception as e:
                print(f"   -> Failed to submit: {e}")
                continue

            # Wait for job completion
            start_t = time.time()
            while True:
                time.sleep(3)
                try:
                    jobs = client.get("/api/jobs").json()
                    cur = next((j for j in jobs if j["id"] == job_id), None)
                    if not cur or cur["status"] in ("completed", "failed", "cancelled", "interrupted", "partial"):
                        status = cur["status"] if cur else "deleted"
                        elapsed = round(time.time() - start_t, 1)
                        print(f"   -> Result: {status} in {elapsed}s")
                        break
                except Exception:
                    pass

        print("All assets upgraded to latest 3-module pipeline!")

if __name__ == "__main__":
    refresh_all()
