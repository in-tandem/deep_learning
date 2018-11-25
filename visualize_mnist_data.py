import numpy as np 
from matplotlib import pyplot as plot 
from load_mnist import load_from_pickled_form, pickled_form


def seeDigits(number_of_rows = 10, number_of_cols = 10, kind = 'train'):
    """
        :param number_of_rows: the number of rows in the plot
        :param number_of_cols: the number of cols in the plot

        there will be 10 * 10 = 100 images plotted in respective 
        subplots

    """
    figures, axis = plot.subplots(nrows = number_of_rows, ncols = number_of_cols, sharex = True, sharey = True)

    data = load_from_pickled_form(path = pickled_form)

    x = data.get('x_'+kind)
    y = data.get('y_'+kind)

    axis = axis.flatten()
    print(axis.shape)
    number_of_images = number_of_rows * number_of_cols

    for i in range(number_of_images):
        
        image = x[y == i%10][0].reshape(28,28)
        axis[i].imshow(image, cmap = 'Greys')

    plot.tight_layout()
    plot.show()


seeDigits()    