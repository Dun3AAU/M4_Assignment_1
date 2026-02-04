#importin packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse

def nn(x, weight):
    #simple neural network function
    return x * weight

def loss(y, y_pred):
    #mean squared error loss function
    return np.mean((y - y_pred) ** 2)

def gradient(x, y, y_pred):
    #compute gradient
    return np.mean(2 * x * (y_pred - y))

def update_weight(weight, grad, learning_rate):
    #update weight using gradient descent
    return weight - learning_rate * grad


def main():

    #argparse for command line arguments, batch or stochastic
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['batch', 'stochastic'], default='batch', help='Training mode: batch or stochastic')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='Learning rate for weight updates')
    parser.add_argument('--weight', type=float, default=10, help='Initial weight value')
    args = parser.parse_args()


    #import data from CSV into dataframe
    df = pd.read_csv('https://raw.githubusercontent.com/aaubs/ds-master/main/data/Swedish_Auto_Insurance_dataset.csv')
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(data_scaled, columns=df.columns)

    #Just to check add newline after
    print(df.head())
    print()

    #defining weight, learning rate and epochs
    weight = args.weight
    learning_rate = args.learning_rate
    epochs = args.epochs

    w_history = []
    w_history.append(weight)
    for i in range(epochs):
        if args.mode == 'batch':
            #batch gradient descent
            X = df['X']
            Y = df['Y']
            #forward pass
            Y_pred = nn(X, weight)
            #feed forward evaluation
            current_loss = loss(Y, Y_pred)
            #backward pass / gradient computation
            grad = gradient(X, Y, Y_pred)
            #weight update / backpropagation
            weight = update_weight(weight, grad, learning_rate)

            #store weight history
            w_history.append(weight)
            print(f'Epoch {i+1}, Loss: {current_loss}, Weight: {weight}')
        else:
            #stochastic gradient descent
            for index, row in df.iterrows():
                X = row['X']
                Y = row['Y']
                #forward pass
                Y_pred = nn(X, weight)
                #feed forward evaluation
                current_loss = loss(Y, Y_pred)
                #backward pass / gradient computation
                grad = gradient(X, Y, Y_pred)
                #weight update / backpropagation
                weight = update_weight(weight, grad, learning_rate)
                
                print(f' Epoch {i+1} | Old Weight: {w_history[i]:.4f}, X: {X:.4f}, Y: {Y:.4f}, Y_pred: {Y_pred:.4f}, Grad: {grad:.4f}, New Weight: {weight:.4f}')

            #store weight history
            w_history.append(weight)
            print(f'Epoch {i+1}, Loss: {current_loss}, Weight: {weight}')





if __name__ == "__main__":
    main()
