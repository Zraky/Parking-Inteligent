import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from PIL import Image, ImageOps


def initialize_parameters(neuron_list):
    parameters = {}
    for i in range(1, len(neuron_list)):
        parameters[f'W{i}'] = np.random.randn(neuron_list[i], neuron_list[i - 1])
        parameters[f'b{i}'] = np.zeros((neuron_list[i], 1))
    return parameters


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def forward_propagation(X, parameters):
    A = X
    activations = {'A0': X}
    for i in range(1, len(parameters) // 2 + 1):
        Z = np.dot(parameters[f'W{i}'], A) + parameters[f'b{i}']
        A = sigmoid(Z)
        activations[f'A{i}'] = A
    return activations


def backward_propagation(X, y, parameters, activations):
    gradients = {}
    m = X.shape[1]
    A_last = activations[f'A{len(parameters) // 2}']
    dZ_last = A_last - y

    for i in reversed(range(1, len(parameters) // 2 + 1)):
        A_prev = activations[f'A{i-1}']
        dW = 1 / m * np.dot(dZ_last, A_prev.T)
        db = 1 / m * np.sum(dZ_last, axis=1, keepdims=True)
        dZ_prev = np.dot(parameters[f'W{i}'].T, dZ_last) * A_prev * (1 - A_prev)

        gradients[f'dW{i}'] = dW
        gradients[f'db{i}'] = db

        dZ_last = dZ_prev

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    for i in range(1, len(parameters) // 2 + 1):
        parameters[f'W{i}'] -= learning_rate * gradients[f'dW{i}']
        parameters[f'b{i}'] -= learning_rate * gradients[f'db{i}']
    return parameters


def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred >= 0.5).astype(int)
    return np.mean(y_pred_binary == y_true)


def neural_network(X, y, neuron_list, learning_rate=0.1, n_iter=1000):
    np.random.seed(0)
    parameters = initialize_parameters(neuron_list)

    train_loss = []
    train_acc = []

    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parameters)
        y_pred = activations[f'A{len(parameters) // 2}']

        loss = compute_loss(y, y_pred)
        accuracy = compute_accuracy(y, y_pred)

        gradients = backward_propagation(X, y, parameters, activations)
        parameters = update_parameters(parameters, gradients, learning_rate)

        train_loss.append(loss)
        train_acc.append(accuracy)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()

    return parameters


def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    return activations

def neural_network(X, y, neuron_list, learning_rate=0.1, n_iter=10000):
    np.random.seed(0)
    parameters = initialize_parameters(neuron_list)

    train_loss = []
    train_acc = []

    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parameters)
        y_pred = activations[f'A{len(parameters) // 2}']

        loss = compute_loss(y, y_pred)
        accuracy = compute_accuracy(y, y_pred)

        gradients = backward_propagation(X, y, parameters, activations)
        parameters = update_parameters(parameters, gradients, learning_rate)

        train_loss.append(loss)
        train_acc.append(accuracy)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()

    return parameters

def image_learning(neuron , total_img, image_reconise, image_random="image_random", extension="jpg"):
    X = []
    y = []
    images = 0
    other = 0
    total_samples = total_img

    for _ in tqdm(range(total_samples)):
        if random.randint(1, 2) != 1 and images < total_samples / 2:
            images += 1
            number = str(images).zfill(5)
            img_path = os.path.join(image_reconise, f"{number}.{extension}")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = ImageOps.grayscale(img)
                img = img.resize((64, 64))
                y.append(1)
            else:
                print(f"File not found: {img_path}")
                continue
        elif other < total_samples / 2:
            other += 1
            number = str(other).zfill(5)
            img_path = os.path.join(image_random, f"{number}.{extension}")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = ImageOps.grayscale(img)
                img = img.resize((64, 64))
                y.append(0)
            else:
                print(f"File not found: {img_path}")
                continue
        else:
            continue

        img = np.array(img)
        img = img.flatten() / 255.0
        X.append(img)

    min_length = min(len(X), len(y))
    X = X[:min_length]
    y = y[:min_length]

    X = np.array(X).T
    y = np.array(y).reshape(1, -1)


    if X.ndim == 1:
        X = X.reshape(1, -1)

    print('dimensions de X:', X.shape)
    print('dimensions de y:', y.shape)

    plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
    neuron_list = [X.shape[0]]
    for i in range(len(neuron)):
        neuron_list.append(neuron[i])
    trained_parameters = neural_network(X, y, neuron_list)
    W_last = trained_parameters[f'W{len(trained_parameters) // 2}']
    b_last = trained_parameters[f'b{len(trained_parameters) // 2}']
    print("Number of neurons in the last layer:", W_last.shape[0])
    print(b_last >= 0.5)
    with open(f"trained_parameter_2_{image_reconise}.pkl", "wb") as f:
        pickle.dump(trained_parameters, f)

neuron = [256, 128, 64, 32, 16, 8, 4, 2, 1]

if __name__ == '__main__':
    image_learning(neuron, 7935 * 2, "cars_train")