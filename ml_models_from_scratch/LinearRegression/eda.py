import pandas as pd

def eda(X: pd.DataFrame, y: pd.DataFrame):
    print(f"Feature set shape : {X.shape}")
    print(f"Target set shape : {y.shape}")
    print("=" * 50)

    print("\nNull Values")
    print(X.isna().sum())
    print(y.isna().sum())

    print("=" * 50)

    print("\nColumns")
    print(X.columns.tolist())

    X.columns = (
        X.columns
        .str.replace(r"^X\d+\s*", '', regex=True)
        .str.replace(' ', '-')

    )

    print("\nColumns after removing prefix")
    print(X.columns.tolist())

    print("=" * 50)

    print("\nInfo")
    print(X.info())
    print(y.info())

    

