#!/usr/bin/env python3
import argparse, csv, os, re, sys

PATIENT_RE = re.compile(r"(patient_\d+)", re.IGNORECASE)

def parse_patient_id(filename):
    m = PATIENT_RE.search(filename)
    return m.group(1).lower() if m else os.path.splitext(filename)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.video_dir):
        sys.exit(f"ERROR: {args.video_dir} not found")

    videos = sorted(
        f for f in os.listdir(args.video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not videos:
        sys.exit("ERROR: no videos found")

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    rows = [
        {
            "video_path": os.path.join(args.video_dir, v),
            "patient_id": parse_patient_id(v),
            "video_id": os.path.splitext(v)[0],
        }
        for v in videos
    ]

    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "patient_id", "video_id"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} videos, {len({r['patient_id'] for r in rows})} patients")

if __name__ == "__main__":
    main()