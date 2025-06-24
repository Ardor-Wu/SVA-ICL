import os
import argparse
import pandas as pd


def main() -> None:
    """Filter dataset by CVE year range."""

    parser = argparse.ArgumentParser(description="Filter dataset by CVE year")
    parser.add_argument(
        "--src",
        default="dataset2",
        help="Path to the dataset containing train/valid/test splits",
    )
    parser.add_argument(
        "--dest",
        default="dataset2_filtered",
        help="Directory where the filtered dataset will be written",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=2011,
        help="Earliest year (inclusive) to exclude from the dataset",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=2019,
        help="Latest year (inclusive) to exclude from the dataset",
    )

    args = parser.parse_args()

    src_dir = args.src
    dest_dir = args.dest
    year_min = args.year_min
    year_max = args.year_max

    splits = ["train", "valid", "test"]

    os.makedirs(dest_dir, exist_ok=True)

    for split in splits:
        src_split = os.path.join(src_dir, split)
        dst_split = os.path.join(dest_dir, split)
        os.makedirs(dst_split, exist_ok=True)

        info_path = os.path.join(src_split, f"{split}_all_infos.xlsx")
        if not os.path.exists(info_path):
            print(f"Warning: {info_path} not found. Skipping {split} split.")
            continue

        df_info = pd.read_excel(info_path)
        df_info["year"] = df_info["cve_id"].str.extract(r"CVE-(\d{4})-")[0].astype(int)
        mask = (df_info["year"] < year_min) | (df_info["year"] > year_max)
        df_filtered = df_info[mask].drop(columns=["year"])

        df_filtered.to_excel(os.path.join(dst_split, f"{split}_all_infos.xlsx"), index=False)

        idx = df_filtered.index

        def filter_csv(name: str) -> None:
            path = os.path.join(src_split, name)
            if not os.path.exists(path):
                return
            df = pd.read_csv(path, header=None)
            df_filtered_csv = df.loc[idx]
            df_filtered_csv.to_csv(os.path.join(dst_split, name), index=False, header=False)

        filter_csv(f"{split}_code.csv")
        filter_csv(f"{split}_ast.csv")
        filter_csv(f"{split}_desc.csv")
        filter_csv(f"{split}_bscore.csv")
        filter_csv(f"{split}_bseveritys.csv")

        all_xlsx = os.path.join(src_split, f"{split}_all.xlsx")
        if os.path.exists(all_xlsx):
            df = pd.read_excel(all_xlsx)
            df_filtered_xlsx = df.loc[idx]
            df_filtered_xlsx.to_excel(os.path.join(dst_split, f"{split}_all.xlsx"), index=False)


if __name__ == "__main__":
    main()
