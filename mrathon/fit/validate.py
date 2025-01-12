
import matplotlib.pyplot as plt

def plotvalidate(w, y, wfit, yfit, ax=None, flatten=True, title='Vector Fitting Validation'):

    if flatten:
        y = y.flatten('F').reshape((-1,9),order='F')
        yfit = yfit.flatten('F').reshape((-1,9),order='F')

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title(title)

    # Plot 'Original' Data as Lines
    ax.plot(w, y)

    # Plot 'Fitted' Data as Scatter
    for yf in yfit.T:
        ax.scatter(wfit, yf, c='r', s=4, zorder=3)

    # X-Axis
    ax.set_xscale('log')
    ax.set_xlabel("$\omega$")

    return ax
