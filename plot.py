import matplotlib.pyplot as plt


def plot_loss(loss):
    plt.figure(0)
    plt.title("Loss")
    plt.xlabel("iteration")
    plt.plot(loss)
    plt.show()


def plot_accuracy(predicted, target):
    plt.figure(0)
    plt.title("Predicted vs True")
    plt.plot(target, color="black", label="Actual")
    plt.plot(predicted, color="green", label="Predicted")
    plt.show()
