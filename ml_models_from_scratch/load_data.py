from ucimlrepo import fetch_ucirepo


def load_data(id: int):
    dataset = fetch_ucirepo(id = id)

    name = dataset.metadata.name
    X = dataset.data.features
    y = dataset.data.targets

    return name, X, y