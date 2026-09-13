"""
Regenerates examples.json from the ACTUAL running API using the
shipped model.pkl/vectorizer.pkl, instead of hand-written/guessed values.
Run this after training so the documented examples always match real output.

Usage: python generate_examples.py
"""
import threading, time, json, urllib.request
import app as flask_app_module

def run():
    flask_app_module.app.run(host="127.0.0.1", port=5099, use_reloader=False)

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(2)

messages = [
    "Congratulations! You won a free iPhone. Claim your reward now.",
    "Your bank account is locked. Verify your password and OTP immediately.",
    "Can you send me the notes after class?",
    "Your parcel is on hold. Click the verification link and pay the delivery fee.",
    "Let's catch up over coffee this weekend."
]

results = []
for m in messages:
    req = urllib.request.Request(
        "http://127.0.0.1:5099/predict",
        data=json.dumps({"message": m}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        response = json.loads(r.read().decode())
    results.append({"request": {"message": m}, "response": response})

with open("examples.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote {len(results)} examples to examples.json")
