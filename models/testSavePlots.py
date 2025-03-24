import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# customizing runtime configuration stored
# in matplotlib.rcParams
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

fig1 = plt.figure()
plt.plot([17, 45, 7, 8, 7], color='orange')

fig2 = plt.figure()
plt.plot([13, 25, 1, 6, 3], color='blue')

Fig3 = plt.figure()
plt.plot([22, 11, 2, 1, 23], color='green')


def save_image(filename):
    # PdfPages is a wrapper around pdf
    # file so there is no clash and create
    # files with no error.
    p = PdfPages(filename)

    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    # iterating over the numbers in list
    for fig in figs:
        # and saving the files
        fig.savefig(p, format='pdf')

        # close the object
    p.close()


# name your Pdf file
filename = "multi_plot_image.pdf"

# call the function
save_image(filename)