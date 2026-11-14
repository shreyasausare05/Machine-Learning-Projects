from sklearn.datasets import load_iris


def main():
    print("-"*30)
    print("Iris classification case study")
    print("-"*30)

    dataset = load_iris()
    #Meta deta of data set

    for i in range(len(dataset.target)):
        print("ID %d, Features %s, Label %s" %(i,dataset.data[i],dataset.target[i]))


if __name__ == "__main__":
    main()