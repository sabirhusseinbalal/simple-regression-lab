from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_model(dataset):
    x = dataset[["cgpa"]]
    y = dataset["package"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    return {
        "model": lr,
        "train_score": lr.score(x_train, y_train),
        "test_score": lr.score(x_test, y_test)
}