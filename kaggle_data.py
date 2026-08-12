from pathlib import Path
import pandas as pd

print("pandas:", pd.__version__)

ARTIFACTS = Path("artifacts")
OUT = Path("artifacts_compat")

OUT.mkdir(exist_ok=True)

files = [
    "X_train.pkl",
    "y_train.pkl",
    "X_test.pkl",
]

for filename in files:
    src = ARTIFACTS / filename

    print(f"Loading: {src}")

    df = pd.read_pickle(src)

    # pandas 3.x StringDtype 기반 column index를
    # 버전 호환성이 좋은 object index로 변환
    df.columns = pd.Index(
        [str(c) for c in df.columns],
        dtype=object,
    )

    dst = OUT / filename.replace(
        ".pkl",
        ".parquet",
    )

    df.to_parquet(
        dst,
        engine="pyarrow",
        index=True,
    )

    print(
        f"Saved: {dst}",
        f"shape={df.shape}",
    )

print("Done.")