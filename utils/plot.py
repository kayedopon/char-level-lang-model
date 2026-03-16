import matplotlib.pyplot as plt


def plot_results(res):
    fig, axis = plt.subplots(1, 2)
    axis[0].plot(res["train_acc"], label="train")
    axis[0].plot(res["test_acc"], label="test")
    axis[0].set_title("Train vs test acc")
    axis[0].legend()

    axis[1].plot(res["train_loss"], label="train")
    axis[1].plot(res["test_loss"], label="test")
    axis[1].set_title("Train vs test loss")
    axis[1].legend()

    plt.show()