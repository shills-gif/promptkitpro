from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]

IN_TXT = ROOT / "PromptKitPro_250_ModelAware.txt"
OUT_CSV = ROOT / "app" / "data" / "prompt_library_v3_250.csv"

text = IN_TXT.read_text(encoding="utf-8")

# Split on each prompt header
blocks = re.split(r"\n(?=PROMPT\s+\d+\s+—\s+)", text)

rows = []

for block in blocks:
    block = block.strip()
    if not block.startswith("PROMPT "):
        continue

    lines = block.splitlines()
    header = lines[0]

    match = re.match(r"PROMPT\s+(\d+)\s+—\s+(.+)", header)
    if not match:
        continue

    prompt_id = match.group(1)
    category = match.group(2)

    prompt_body = "\n".join(lines[1:]).strip()

    rows.append({
        "id": prompt_id,
        "category": category,
        "prompt": prompt_body
    })

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "category", "prompt"],
        quoting=csv.QUOTE_ALL
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"SUCCESS: Wrote {len(rows)} prompts to {OUT_CSV}")
