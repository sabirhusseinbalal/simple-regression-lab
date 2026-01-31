import pandas as pd
from load_data import load_data
from train import train_model
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    dataset = load_data()

    # plt.figure(figsize=(5,3))
    # sns.scatterplot(x="cgpa",y="package",data=dataset)
    # plt.show()

    results = train_model(dataset)
    lr = results["model"]
    test_data = pd.DataFrame([[6.89]], columns=["cgpa"])
    print("Prediction for CGPA 6.89:", lr.predict(test_data))
    print(results)


    plt.figure(figsize=(5,3))
    sns.scatterplot(x="cgpa",y="package",data=dataset)
    plt.plot(dataset["cgpa"],lr.predict(dataset[["cgpa"]]),c="red")
    plt.legend(["org data", "predict line"])
    plt.show()



if __name__ == "__main__":
    main()