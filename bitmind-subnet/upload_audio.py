"""Upload audio model with increased timeout."""
import json
import hashlib
import os
import requests
import bittensor as bt
from gas.protocol.epistula import generate_header


AUDIO_ZIP = "/root/audio_detector.zip"
UPLOAD_ENDPOINT = "https://upload.bitmind.ai/upload"

wallet = bt.Wallet(name="miner", hotkey="default")

# Compute hash
print("Computing file hash...")
h = hashlib.sha256()
file_size = os.path.getsize(AUDIO_ZIP)
with open(AUDIO_ZIP, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        h.update(chunk)
file_hash = h.hexdigest()
print(f"Hash: {file_hash}")
print(f"Size: {file_size} bytes ({file_size/1024/1024:.1f} MB)")

# Step 1: Get presigned URL
print("\n[1/3] Getting presigned URL...")
payload = {
    "filename": "audio_detector.zip",
    "file_size": file_size,
    "expected_hash": file_hash,
    "content_type": "application/octet-stream",
    "modality": "audio",
}
payload_json = json.dumps(payload, separators=(",", ":"))
payload_bytes = payload_json.encode("utf-8")

headers = generate_header(wallet.hotkey, payload_bytes)
headers["Content-Type"] = "application/json"

resp = requests.post(
    f"{UPLOAD_ENDPOINT}/presigned",
    data=payload_bytes,
    headers=headers,
    timeout=30,
)
result = resp.json()
print(f"Response: {json.dumps(result, indent=2)}")

if resp.status_code != 200:
    print(f"Failed to get presigned URL: {result}")
    exit(1)

presigned_data = result["data"]
model_id = presigned_data["model_id"]
presigned_url = presigned_data["presigned_url"]
r2_key = presigned_data.get("r2_key", "")
print(f"Model ID: {model_id}")
print(f"R2 Key: {r2_key}")

# Step 2: Upload to R2 with increased timeout
print("\n[2/3] Uploading to R2 (timeout: 10 min)...")
with open(AUDIO_ZIP, "rb") as f:
    file_content = f.read()

upload_resp = requests.put(
    presigned_url,
    data=file_content,
    headers={"Content-Type": "application/octet-stream"},
    timeout=600,  # 10 minutes
)
print(f"Upload status: {upload_resp.status_code}")
if upload_resp.status_code != 200:
    print(f"Upload failed: {upload_resp.text}")
    exit(1)
etag = upload_resp.headers.get("ETag", "")
print(f"ETag: {etag}")

# Step 3: Confirm upload
print("\n[3/3] Confirming upload...")
confirm_payload = {
    "model_id": model_id,
    "file_hash": file_hash,
}
confirm_json = json.dumps(confirm_payload, separators=(",", ":"))
confirm_bytes = confirm_json.encode("utf-8")
confirm_headers = generate_header(wallet.hotkey, confirm_bytes)
confirm_headers["Content-Type"] = "application/json"

confirm_resp = requests.post(
    f"{UPLOAD_ENDPOINT}/confirm",
    data=confirm_bytes,
    headers=confirm_headers,
    timeout=30,
)
print(f"Confirm status: {confirm_resp.status_code}")
print(f"Confirm response: {confirm_resp.text}")

if confirm_resp.status_code == 200:
    print(f"\nAudio model uploaded successfully! Model ID: {model_id}")
else:
    print(f"\nConfirmation failed!")
