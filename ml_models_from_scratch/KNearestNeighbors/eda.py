def eda(X, y):
    print("=" * 50)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("=" * 50)
    print("Class Distribution")
    print(y.value_counts())
    print("=" * 50)

    print("Null Values")
    print(X.isna().sum())
    print(y.isna().sum())
    print("=" * 50)

    print("=" * 50)
    print("Columns")
    print(list(X.columns))

    X.columns = X.columns.str.replace(' ', '-')
    print("Columns after")
    print(list(X.columns))

    print("=" * 50)
    print("Info")
    print(X.info())
    print("=" * 50)

