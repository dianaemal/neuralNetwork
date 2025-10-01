import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd
class NeuralNetwork:
    def __init__(self,  input_size=784, output_size = 10, learning_rate=0.0005):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layers = [784, 128, 64,  10]
        self.weights = []
        self.biases = []
        self.gradients_w = []
        self.gradients_b = []
        self.iterations = 3
        self.epochs = 10
        self.loss_history = []
        self.accuracy_history = []
        
      
    def initialize_parameters(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            #* np.sqrt(2.0 / self.layers[i]
            bias = np.zeros((1, self.layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
        

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return np.where(Z>0, 1, 0)
    
    def softmax(self, Z):
        # to prevent overflow, we subtract the max from each row
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        softmax = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return softmax
    
    def forward_propagation(self, X):
        self.Z = []
        self.A = [X]

        for i in range(len(self.weights)):
            z = np.dot(self.A[i], self.weights[i]) + self.biases[i]

            #if np.isnan(z).any() or np.isinf(z).any():
               # print(f"NaN or Inf detected in layer {i} forward pass")
            self.Z.append(z)

            if i == (len(self.layers) - 1):
                a = self.softmax(z)
            else:
                a = self.relu(z)

            self.A.append(a)

        return self.A, self.Z
    
    def one_hot_encode(self, y):
        # create a zero matrix of shape (number of samples, number of classes)
        ##encoded_array = np.zeros((y.size, y.max() + 1), dtype=int)
        encoded_array = np.zeros((y.size, self.output_size), dtype=int)
        # set the appropriate elements to 1
        # np.arrange creates an array of indices from 0 to y.size-1
        # y contains the class labels for each sample
        # example: if y = [1, 0, 3], then encoded_array will be:
        # [[0, 1, 0, 0],
        #  [1, 0, 0, 0],
        #  [0, 0, 0, 1]]
        encoded_array[np.arange(y.size), y] = 1
        return encoded_array
    
    def compute_loss(self, y_true, y_pred):
        # cross-entropy loss
        n = y_true.shape[0]
        # adding a small value to avoid log(0)
       # loss = -np.sum(y_true * np.log(y_pred + 1e-9)) 
       # return loss / n
    
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss 
    

    def backward_propagation(self, X, y):

        y_true = self.one_hot_encode(y)

        A, Z = self.forward_propagation(X)
        y_pred = A[-1]

        # compute the gradient of the loss with respect to the output
        dZ = y_pred - y_true
        dL_dW_3 = np.dot(A[-2].T, dZ)
        # we sum across the rows to get the gradient for each bias
        dL_db_3 = np.sum(dZ, axis=0, keepdims=True)
        # clipping gradients to prevent exploding gradients
        dL_dW_3 = np.clip(dL_dW_3, -1, 1)
        dL_db_3 = np.clip(dL_db_3, -1, 1)
        self.gradients_w.append(dL_dW_3)
        self.gradients_b.append(dL_db_3)

        # update Weights and biases for the output layer
        self.weights[-1] -= self.learning_rate * dL_dW_3
        self.biases[-1] -= self.learning_rate * dL_db_3

        # compute the gradient of the loss with respect to the hidden layer

        for i in range (len(self.weights) - 2, -1, -1):
            dZ = np.dot(dZ, self.weights[i+1].T) * self.relu_derivative(Z[i])
            dL_dW = np.dot(A[i].T, dZ)
            # clipping gradients to prevent exploding gradients
            dL_dW = np.clip(dL_dW, -1, 1)

            dL_db = np.sum(dZ, axis=0, keepdims=True)
            dL_db = np.clip(dL_db, -1, 1)
            self.gradients_w.append(dL_dW)
            self.gradients_b.append(dL_db)  
            # update Weights and biases for the hidden layers
            self.weights[i] -= self.learning_rate * dL_dW
            self.biases[i] -= self.learning_rate * dL_db


    
    def batch_generator(self, X, Y, batch_size):
        n = X.shape[0]
        batches_x = []
        batches_y = []
        for i in range(0, n, batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = Y[i:i+batch_size]
            batches_x.append(batch_x)
            batches_y.append(batch_y)
        return batches_x, batches_y
                        
        
    def train(self, X, Y):
        batches_x, batches_y = self.batch_generator(X, Y, batch_size=200)
        self.initialize_parameters()
        for i in range(self.epochs):
            for k in range(len(batches_x)):
                for j in range (self.iterations):
                    preds, _ = self.forward_propagation(batches_x[k])
                    acc = self.accuracy(batches_y[k], np.argmax(preds[-1], axis=1))
                    self.accuracy_history.append(acc)
                    loss = self.compute_loss(self.one_hot_encode(batches_y[k]), preds[-1])
                    self.loss_history.append(loss)
                    self.backward_propagation(batches_x[k], batches_y[k])

                    # Print progress every 100 iterations
                    if k % 100 == 0:
                        print(f"Epoch {i+1}, Batch {k+1}, Iteration {j+1}, Loss: {loss:.4f}")

                   
    def predict(self, X):
        A, _ = self.forward_propagation(X)
        # argmax to get the index of the highest probability
        predictions = np.argmax(A[-1], axis=1)
        return predictions
    
    def draw_loss_curve(self):
       
        
       # Plotting loss and accuracy curves in one graph
        plt.plot(self.loss_history, label='Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.savefig("loss.png", dpi=300, bbox_inches="tight")
        plt.show()

    def draw_accuracy_curve(self):
        plt.plot(self.accuracy_history, label='Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.savefig("accuracy.png", dpi=300, bbox_inches="tight")
        plt.show()
        

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
   
        
    



    

    
          


   



      