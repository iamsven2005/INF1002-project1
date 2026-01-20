import pandas as pd
import glob

# Find all CSV files that match the pattern:
# Reviews_part_1.csv, Reviews_part_2.csv, Reviews_part_3.csv, etc.
# glob returns a list of matching filenames.
# sorted() ensures the files are processed in correct order.
parts = sorted(glob.glob("Reviews_part_*.csv"))

# Read each CSV file and combine them into a single DataFrame
# pd.read_csv(p) reads one CSV file
# The generator expression creates a list of DataFrames
# pd.concat merges them vertically (row-wise)
# ignore_index=True resets the row index so it runs from 0...N continuously
df = pd.concat(
    (pd.read_csv(p) for p in parts),
    ignore_index=True
)

# Save the merged DataFrame into a new CSV file called Reviews.csv
# index=False prevents pandas from writing the DataFrame index as a column
df.to_csv("Reviews.csv", index=False)

# Print a confirmation message showing how many files were merged
print(f"Merged {len(parts)} files into Reviews.csv")
