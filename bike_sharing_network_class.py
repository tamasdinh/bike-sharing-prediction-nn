import numpy as np

class NeuralNetwork(object):
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # setting number of nodes in layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # initializing weights with normal distribution
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        # activation function for hidden layer (sigmoid)
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))


    def train(self, features, targets):
        '''
        Implements mini-batch training strategy. 
        Arguments:
            - features: 2D array, each row is one data record, each column is a feature
            - targets: 1D array of target values
        '''
        n_records = features.shape[0]
        
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        '''
        Implements forward pass 
        Arguments - X: features batch
        '''

        # signals into hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) 
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # signals from final output layer (predictions)
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs


    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''
        Implements backpropagation.
        Arguments:
            - final_outputs: output from forward pass
            - y: target (i.e. label) batch
            - delta_weights_i_h: change in weights from input to hidden layers
            - delta_weights_h_o: change in weights from hidden to output layers
        '''

        error = (y - final_outputs)
        output_error_term = error
        hidden_error_term = np.dot(output_error_term, self.weights_hidden_to_output.T) * hidden_outputs * (1 - hidden_outputs)
        
        # Weight steps - incremented due to mini-batch approach
        delta_weights_i_h += hidden_error_term * X[:, None]
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        
        return delta_weights_i_h, delta_weights_h_o


    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        '''
        Updates weights on gradient descent step.
        Arguments:
            - delta_weights_i_h: change in weights from input to hidden layers
            - delta_weights_h_o: change in weights from hidden to output layers
            - n_records: number of records (due to mini-batch approach, this variables is used to average gradient steps derived from individual records)
        '''

        self.weights_hidden_to_output += self.lr * delta_weights_h_o.reshape(self.hidden_nodes, 1) / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records


    def run(self, features):
        '''
        Runs a forward pass through the network (for testing / prediction purposes).
        Arguments:
            - features: 1D array of feature values
        '''
        
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs