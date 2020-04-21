import matplotlib.pyplot as plt
import numpy

def get_general_plot(
        x,
        y,
        xLabel = None,
        yLabel = None,
        title = None,
        savefile = None
):

    fig = plt.figure()
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.plot([1, 2, 3], [3, 4, 5], 'r-')
    if title is not None:
        plt.title(title)
    if xLabel is not None:
        plt.xlabel(xLabel)

    if yLabel is not None:
        plt.ylabel(yLabel)

    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    return fig

