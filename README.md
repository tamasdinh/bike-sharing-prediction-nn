# Bike sharing demand predictions with a feed-forward neural network

In this project, I built a simple feed-forward neural network to predict daily bike rental ridership, based on a public UCI Machine Learning dataset.\
\
Neural networks are extremely powerful in finding complex patterns in datasets. The feedforward neural network consists of a number of fully connected layers:
- the input layer consists of the variables used for the prediction task
- an optional number of "hidden" layers, each of which analyzes the patterns of the data from the previous layer
- an output layer that combines the results of the last hidden layer and produces the prediction (classification or regression)

The layers in the neural network being fully connected means that every node in every hidden layer processes information from all nodes from the previous layer. In the first hidden layer, this means processing the input data; in subsequent hidden layers, this means processing, combining data from previous hidden layers. Then when this is all done, the network checks its accuracy against known samples and backpropagates prediction error, gradually changing the weighting between nodes in the network until ample accuracy is achieved. This is how a neural network can identify complex relationships in datasets.\
\
In this particular case, the dataset presented a quite complex time series with several, only partially cyclical fluctuations. I utilized only 1 hidden layer but with 20 nodes to capture the variability of demand. In the end, as can be seen in the notebook, the prediction is quite accurate for most of the time periods, with a slight weakness around the Festive Season. This is due to the fact that the training data didn't include much information for previous Festive Seasons and as such, couldn't train properly for this scenario.

## Files
This was a small experimental project, and as such, it contains 2 associated files:
- the IPython notebook [_Bike-sharing-predictions-nn.ipynb_](Bike-sharing-predictions-nn.ipynb) (viewable from GitHub's interface) contains data preparation and analysis
- the Python executable [*bike_sharing_network_class.py*](bike_sharing_network_class.py) (also viewable from GitHub's interface) contains the definition of the neural network object used for the predictions (implemented in raw NumPy).

Datasets can be found in the [/Assets](./Assets) folder.

For more details, please see the two files above.

## Built With

Just the usual basic tools:

* [Python3](https://www.python.org) - Good Old Python :)
* [Project Jupyter](https://jupyter.org) - The go-to notebook interface for interactive Python programming and documentation
* [Pandas](https://pandas.pydata.org) - The Python Data Analysis Library
* [NumPy](http://www.numpy.org) - The Numerical Python Library
* [MS Visual Studio Code](http://code.visualstudio.com) - My go-to IDE for scripting


## Authors

* **Tamas Dinh** [LinkedIn profile](https://www.linkedin.com/in/tamasdinh/)