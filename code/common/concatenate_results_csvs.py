import os
import pandas as pd


def find_csv_files(root_path):
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def read_and_clean_csv(file_path, column_name):
    # Try a couple of common encodings; you can add more if needed.
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            df.drop(column_name, axis=1, inplace=True, errors="ignore")
            df.dropna(how="all", inplace=True)
            return df, None
        except Exception as e:
            last_err = e
    return None, last_err


def main():
    root_paths = [r"G:\Stacked\Results\gwas_diversity_results"]
    column_to_remove = "0"
    output_file = r"G:\Stacked\Results\ResNet_upth0.95_downth0.3_Jan26_23-15-35_2026_Mini_GWAS_dataset.csv"

    all_csv_files = []
    for path in root_paths:
        all_csv_files.extend(find_csv_files(path))

    print(f"Found {len(all_csv_files)} CSV files.")

    dfs = []
    failures = []
    total_rows = 0

    for i, fp in enumerate(all_csv_files, 1):
        df, err = read_and_clean_csv(fp, column_to_remove)
        if err is not None:
            failures.append((fp, repr(err)))
            print(f"[{i}/{len(all_csv_files)}] FAIL  {fp}  -> {err}")
            continue

        rows = len(df)
        total_rows += rows
        dfs.append(df)
        print(f"[{i}/{len(all_csv_files)}] OK    {fp}  ({rows} rows)")

    if not dfs:
        raise RuntimeError("No CSVs were successfully read. See failures above.")

    concatenated = pd.concat(dfs, ignore_index=True, sort=False)
    concatenated.to_csv(output_file, index=False)

    print("\n--- Summary ---")
    print(f"Successful files: {len(dfs)}")
    print(f"Failed files:     {len(failures)}")
    print(f"Total rows read (pre-concat): {total_rows}")
    print(f"Final rows written:          {len(concatenated)}")
    print(f"Output: {output_file}")

    if failures:
        # write a failure log next to output
        log_path = os.path.splitext(output_file)[0] + "_failures.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            for fp, err in failures:
                f.write(f"{fp}\t{err}\n")
        print(f"Failure log written: {log_path}")


if __name__ == "__main__":
    main()
