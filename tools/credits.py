"""Licence summary and attribution list for the collected photos.

Every photo keeps its licence, author and source URL in data/metadata.csv. This writes
  data/credits.csv   one line per photo: image_id, title, author, licence, source URL, flags
  results/licences.md  licence counts and what each restriction means for the project
Use credits.csv to credit any photo shown in the report, slides or app screenshots
("Title" by Author, Licence, URL). Flags:
  NC  non-commercial only (fine for a course project; not for a commercial app)
  ND  no derivatives: do not publish modified versions (crops, phone-degraded copies, collages)
  SA  share-alike: published modified versions must keep the same licence
  no_author  author missing in the source metadata (check the source page before crediting)

Usage:  python tools/credits.py
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import DATA, METADATA_CSV, RESULTS, read_csv, write_csv  # noqa: E402


def flags(lic, author):
    f = []
    up = lic.upper()
    if re.search(r"\bNC\b|-NC", up):
        f.append("NC")
    if re.search(r"\bND\b|-ND", up):
        f.append("ND")
    if re.search(r"\bSA\b|-SA", up):
        f.append("SA")
    if not author.strip():
        f.append("no_author")
    return " ".join(f)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0],
                            epilog='See the module docstring for details.').parse_args()
    meta = read_csv(METADATA_CSV)
    out = pd.DataFrame({"image_id": meta.image_id, "title": meta.title.str.removeprefix("File:"),
                        "author": meta.artist, "licence": meta.license, "source_url": meta.url,
                        "flags": [flags(l, a) for l, a in zip(meta.license, meta.artist)]})
    write_csv(out, DATA / "credits.csv")
    counts = meta.license.value_counts()
    nflag = {k: int(out["flags"].str.contains(rf"\b{k}\b").sum()) for k in ("NC", "ND", "SA", "no_author")}
    md = ["# Photo licences\n", f"{len(meta)} photos; per-photo credits in `data/credits.csv`.\n",
          "| Restriction | Photos | Meaning for this project |", "|---|---|---|",
          f"| NC (non-commercial) | {nflag['NC']} | fine for a university course project; must be removed before any commercial use |",
          f"| ND (no derivatives) | {nflag['ND']} | may be used for training; do not publish modified versions (crops, phone-degraded copies, contact sheets) |",
          f"| SA (share-alike) | {nflag['SA']} | published modified versions must carry the same licence |",
          f"| author missing | {nflag['no_author']} | look the author up on the source page before crediting |",
          "\nAny photo shown in the report, slides or app screenshots must be credited: "
          "\"Title\" by Author, Licence, source URL (from `data/credits.csv`). "
          "Do not redistribute `data/raw/` without `data/credits.csv`.\n",
          "| Licence | Photos |", "|---|---|", *[f"| {k} | {v} |" for k, v in counts.items()]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "licences.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:9]))


if __name__ == "__main__":
    main()
