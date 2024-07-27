import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DNN:
    def __init__(self, layer_sizes, nb_classes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.h = [np.zeros((size, 1)) for size in layer_sizes]
        self.a = [np.zeros((size, 1)) for size in layer_sizes[1:]]
        for i in range(1, self.num_layers):
            # self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01)
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2 / layer_sizes[i-1]))
            self.biases.append(np.zeros((layer_sizes[i], 1)))

        self.y = np.zeros(nb_classes)
        

    def forward(self, X):
        """ 
        Computes the output of the network for given input
        Applies weights, biases, and activation function layer by layer
        """
        z = X
        self.h[0] = X
        for ind in range(self.num_layers - 1):  
            z = np.dot(self.weights[ind], z) + self.biases[ind]
            self.a[ind] = z

            #Activation
            z = self.activate(z)  
            self.h[ind+1] = z
        self.y = z
        return self.y
     
    def compute_loss(self, y_pred, y_true):
        """ 
        Compute cross-entropy loss between predicted and true values
        Adds small epsilon to avoid log(0) errors
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    def backward_prop(self,y_pred, y_true , lr=0.05):
        """
        Backpropagation: computes gradients and updates weights/biases
        Implements the chain rule to propagate error backwards through the network

        Resources used: 
            http://chercheurs.lille.inria.fr/pgermain/neurones2019/04-retropropagation.pdf
            https://medium.com/analytics-vidhya/backpropagation-for-dummies-e069410fa585
        """
        g =y_pred-y_true
        m = y_true.shape[1]

        for ind in range(self.num_layers - 2, -1, -1):

            deriv_weights =(1/m) * np.dot(g, np.transpose(self.h[ind]))
            deriv_bias = (1/m) * np.sum(g,axis=1,keepdims=True)
            if ind > 0:
                g = np.dot(np.transpose(self.weights[ind]), g)
                g = g*self.activate_derivative(self.a[ind-1])

            #MAJ biais et poids
            self.weights[ind] -= lr * deriv_weights
            self.biases[ind] -= lr * deriv_bias
    
    def train(self, X, y_true, learning_rate, epochs):
        """
        This function implements a basic, sample-by-sample training approach, 
        eschewing mini-batches for simplicity. While less efficient, 
        it provides a clear, straightforward implementation for learning purposes.        
        """

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(X.shape[1]):
                x = X[:, i:i+1]
                y = y_true[:, i:i+1]
                
                y_pred = self.forward(x)
                self.backward_prop(y_pred, y, learning_rate)
                epoch_loss += self.compute_loss(y_pred, y)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss/X.shape[1]}")
    
    def predict(self, X):
        return self.forward(X)

    def activate(self, z, fct = 'sigmoid'):
        if fct == 'sigmoid':
            return 1 / (1 + np.exp(-z))

    def activate_derivative(self, z, fct = 'sigmoid'):
        if fct == 'sigmoid':
            return self.activate(z) * (1 - self.activate(z))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        
if __name__ == "__main__":
    
    #Test on the moons dataset to verify the neural network's functionality
    #This classic problem helps ensure the implementation works correctly
    
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = np.transpose(np.eye(2)[y_train])
    y_test = np.transpose(np.eye(2)[y_test])

    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    X_train = X_train.T
    X_test = X_test.T


    layer_sizes = [2, 10, 2]
    nn_moons = DNN(layer_sizes, 2)
    nn_moons.train(X_train, y_train, learning_rate=0.1, epochs=100)

    y_pred = nn_moons.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=0)
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Précision sur l'ensemble de test : {accuracy:.4f}")
