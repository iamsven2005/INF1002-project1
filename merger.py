import pandas as pd
import glob

parts = sorted(glob.glob("Reviews_part_*.csv"))

df = pd.concat(
    (pd.read_csv(p) for p in parts),
    ignore_index=True
)

df.to_csv("Reviews.csv", index=False)

print(f"Merged {len(parts)} files into Reviews.csv")
