import pandas as pd

INPUT = "Reviews.csv"
ROWS_PER_CHUNK = 100_000   # adjust as needed
OUTPUT_PREFIX = "Reviews_part"

for i, chunk in enumerate(pd.read_csv(INPUT, chunksize=ROWS_PER_CHUNK)):    # reviews csv files
    chunk.to_csv(f"{OUTPUT_PREFIX}_{i+1}.csv", index=False)

print("Done")