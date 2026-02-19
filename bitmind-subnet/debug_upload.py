#!/usr/bin/env python3
from gas.protocol.model_uploads import generate_presigned_url
import bittensor as bt
import json

wallet = bt.wallet(name="miner", hotkey="default")

file_hash = "d6cfd3f16964dc1eaa32e9243e5ade37e3107a678e1ab85befae2042ddbb6392"
file_size = 522531168

result = generate_presigned_url(
    wallet,
    "https://upload.bitmind.ai/upload",
    "video_detector.zip",
    file_size,
    file_hash,
    "application/octet-stream",
    "video"
)
sc = result["status_code"]
ok = result["success"]
resp = result["response"]
print(f"Status: {sc}")
print(f"Success: {ok}")
print(f"Full response: {json.dumps(resp, indent=2)}")
