import matplotlib.pyplot as plt


def plot_loss(loss, model_name, file_path):
    plt.figure(0)
    plt.title("Loss " + model_name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.plot(loss)
    plt.savefig(file_path)
    plt.show()


def plot_accuracy(predicted, target, model_name, file_path):
    plt.figure(0)
    plt.title("Predicted-True Comparison " + model_name)
    plt.ylabel("Price")
    p = plt.plot(target, color="black", label="True")
    t = plt.plot(predicted, "--", color="green", label="Predicted")
    plt.legend(handles=p)
    plt.legend(handles=t)
    plt.savefig(file_path)
    plt.show()
