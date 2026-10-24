def eda(X, y):
    print(f"Feature set shape : {X.shape}")
    print(f"Target set shape : {y.shape}")
    print("=" * 50)

    print("\nNull Values")
    print(X.isna().sum())
    print(y.isna().sum())

    print("=" * 50)

    print("\nColumns")
    print(X.columns.tolist())

    print("=" * 50)
    print("Class Distribution")
    print(y.value_counts())

