import pickle
import numpy as np
from PIL import Image, ImageOps


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def forward_propagation(X, parameters):
    A = X
    activations = {'A0': X}
    for i in range(1, len(parameters) // 2 + 1):
        Z = np.dot(parameters[f'W{i}'], A) + parameters[f'b{i}']
        A = 1 / (1 + np.exp(-Z))
        activations[f'A{i}'] = A
    return activations


def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    A_last = activations[f'A{len(parametres) // 2}']
    prediction = (A_last >= 0.5).astype(int)
    return A_last


def image_reconise(image, trained_file_pkl):
    with open(f"{trained_file_pkl}", "rb") as f:
        parameters = pickle.load(f)

    img = Image.open(image)
    img = ImageOps.grayscale(img)
    img = img.resize((64, 64))
    img = np.array(img).flatten() / 255.0

    X = img.reshape((1, -1)).T

    result = predict(X, parameters)
    return result


image = ""
if __name__ == '__main__':
    img = image_reconise(image, "trained_parameter_cars.pkl")
    print(f"predict : {img}")
    if img >= 0.5:
        print(f"predict : TRUE")
    else:
        print(f"predict : False")