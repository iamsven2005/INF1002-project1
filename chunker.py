import pandas as pd

INPUT = "Reviews.csv"
ROWS_PER_CHUNK = 100_000   # adjust
OUTPUT_PREFIX = "Reviews_part"

for i, chunk in enumerate(pd.read_csv(INPUT, chunksize=ROWS_PER_CHUNK)):
    chunk.to_csv(f"{OUTPUT_PREFIX}_{i+1}.csv", index=False)

print("Done")