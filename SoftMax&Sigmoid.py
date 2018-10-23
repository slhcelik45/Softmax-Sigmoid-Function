import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x)/float(sum(np.exp(x)))

def sigmoid(xx):
    sigmoid_deger=[1/float(1+np.exp(-x)) for x in xx]
    return sigmoid_deger


"""
def line_graph_sigmoid(x,y,x_title,y_title):

    plt.plot(x,y,c='green')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()
"""
def line_graph(x,y,yy,x_title,y_title):

    plt.plot(x,y,c='red')
    plt.plot(x,yy,c='green')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.text(-1,0.8,"Sigmoid>")
    plt.text(6.3,0.6,"Softmax>")
    plt.show()




graph_x=range(-10,10)
graph_y=softmax(graph_x)
graph_y1=sigmoid(graph_x)


line_graph(graph_x, graph_y,graph_y1, "Inputs", "Scores")
#line_graph_sigmoid(graph_x,graph_y1,"Inputs","Sigmoid Score")
