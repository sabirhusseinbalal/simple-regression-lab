import pandas as pd

def load_data():
    dataset = pd.read_csv("data/students.csv")


    if dataset.isnull().sum().any():
        dataset = dataset.dropna()

    return dataset


if __name__ == "__main__":
    dataset = load_data()
    print(dataset.head())
    print("\nMissing values:\n", dataset.isnull().sum())
