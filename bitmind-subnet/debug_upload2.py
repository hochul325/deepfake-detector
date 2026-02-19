#!/usr/bin/env python3
from gas.protocol.model_uploads import generate_presigned_url
import bittensor as bt
import json

wallet = bt.wallet(name="miner", hotkey="default")

# Test with realistic sizes for each modality
for modality in ["image", "video", "audio"]:
    result = generate_presigned_url(
        wallet,
        "https://upload.bitmind.ai/upload",
        f"{modality}_detector.zip",
        300000000,  # 300MB
        "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "application/octet-stream",
        modality
    )
    sc = result["status_code"]
    detail = result.get("response", {}).get("detail", str(result.get("response", "")))
    print(f"  {modality}: status={sc} - {detail}")
    print()
