from pathlib import Path
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_FILE = Path(__file__).with_name("mnist.npz")
PARAMETER_FILE = Path(__file__).with_name("mnist_cnn_parameters.pt")

NUMBER_OF_FILTERS = 4
FILTER_SIZE = 3
NUMBER_OF_INPUTS = 4 * 13 * 13
NUMBER_OF_CLASSES = 10

EPOCHS = 10
NUMBER_OF_IMAGES = 600
NUMBER_OF_BATCHES = 10
LEARNING_RATE = 0.05
NUMBER_OF_TEST_IMAGES = 100


def load_data():
    if not MNIST_FILE.exists():
        print("Downloading MNIST...")
        urllib.request.urlretrieve(MNIST_URL, MNIST_FILE)

    with np.load(MNIST_FILE) as data:
        x_train = data["x_train"].astype(np.float32) / 255.0
        y_train = data["y_train"]
        x_test = data["x_test"].astype(np.float32) / 255.0
        y_test = data["y_test"]

    # (images, height, width) -> (images, channels, height, width)
    x_train = torch.tensor(x_train[:, None, :, :], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test[:, None, :, :], dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return x_train, y_train, x_test, y_test


def initialize_parameters():
    filters = torch.randn(
        NUMBER_OF_FILTERS,
        1,
        FILTER_SIZE,
        FILTER_SIZE,
    ) * 0.1
    convolution_bias = torch.zeros(NUMBER_OF_FILTERS)

    linear_weights = torch.randn(NUMBER_OF_INPUTS, NUMBER_OF_CLASSES) * 0.01
    linear_bias = torch.zeros(NUMBER_OF_CLASSES)

    filters.requires_grad_()
    convolution_bias.requires_grad_()
    linear_weights.requires_grad_()
    linear_bias.requires_grad_()

    return filters, convolution_bias, linear_weights, linear_bias


def convolution_one_image(image, filters, bias):
    _, image_height, image_width = image.shape
    _, _, filter_height, filter_width = filters.shape

    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    output_rows = []

    for row in range(output_height):
        output_columns = []

        for column in range(output_width):
            patch = image[
                :,
                row : row + filter_height,
                column : column + filter_width,
            ]
            values = (filters * patch).sum(dim=(1, 2, 3)) + bias
            output_columns.append(values)

        output_rows.append(torch.stack(output_columns, dim=1))

    return torch.stack(output_rows, dim=1)


def relu(x):
    output = x.clone()
    output[output < 0] = 0
    return output


def max_pool_2x2(x):
    _, height, width = x.shape
    output_rows = []

    for row in range(0, height, 2):
        output_columns = []

        for column in range(0, width, 2):
            patch = x[:, row : row + 2, column : column + 2]
            values = patch.amax(dim=(1, 2))
            output_columns.append(values)

        output_rows.append(torch.stack(output_columns, dim=1))

    return torch.stack(output_rows, dim=1)


def flatten(x):
    output = []
    channels, height, width = x.shape

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                output.append(x[channel, row, column])

    return torch.stack(output)


def linear(x, weights, bias):
    return x @ weights + bias


def forward(image, filters, convolution_bias, linear_weights, linear_bias):
    output = convolution_one_image(image, filters, convolution_bias)
    output = relu(output)
    output = max_pool_2x2(output)
    output = flatten(output)
    return linear(output, linear_weights, linear_bias)


def train(x_train, y_train, filters, convolution_bias, linear_weights, linear_bias):
    batch_size = NUMBER_OF_IMAGES // NUMBER_OF_BATCHES

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_count = 0

        for batch in range(NUMBER_OF_BATCHES):
            start = batch * batch_size

            for index in range(batch_size):
                image_index = start + index
                scores = forward(
                    x_train[image_index],
                    filters,
                    convolution_bias,
                    linear_weights,
                    linear_bias,
                )
                loss = F.cross_entropy(scores, y_train[image_index])

                total_loss += loss.item()
                (loss / batch_size).backward()

                if scores.argmax() == y_train[image_index]:
                    correct_count += 1

            with torch.no_grad():
                filters -= LEARNING_RATE * filters.grad
                convolution_bias -= LEARNING_RATE * convolution_bias.grad
                linear_weights -= LEARNING_RATE * linear_weights.grad
                linear_bias -= LEARNING_RATE * linear_bias.grad

            filters.grad.zero_()
            convolution_bias.grad.zero_()
            linear_weights.grad.zero_()
            linear_bias.grad.zero_()

        average_loss = total_loss / NUMBER_OF_IMAGES
        accuracy = correct_count / NUMBER_OF_IMAGES
        print(
            "epoch:", epoch,
            "average loss:", average_loss,
            "accuracy:", accuracy,
        )


def evaluate(
    x_test,
    y_test,
    filters,
    convolution_bias,
    linear_weights,
    linear_bias,
    number_of_test_images=NUMBER_OF_TEST_IMAGES,
):
    number_of_test_images = min(number_of_test_images, len(x_test))
    total_loss = 0.0
    correct_count = 0

    with torch.no_grad():
        for index in range(number_of_test_images):
            scores = forward(
                x_test[index],
                filters,
                convolution_bias,
                linear_weights,
                linear_bias,
            )
            loss = F.cross_entropy(scores, y_test[index])

            total_loss += loss.item()
            if scores.argmax() == y_test[index]:
                correct_count += 1

    average_loss = total_loss / number_of_test_images
    accuracy = correct_count / number_of_test_images

    print(
        "test average loss:", average_loss,
        "test accuracy:", accuracy,
    )

    return average_loss, accuracy


def save_parameters(
    filename,
    filters,
    convolution_bias,
    linear_weights,
    linear_bias,
):
    parameters = {
        "filters": filters.detach(),
        "convolution_bias": convolution_bias.detach(),
        "linear_weights": linear_weights.detach(),
        "linear_bias": linear_bias.detach(),
    }
    torch.save(parameters, filename)
    print("saved parameters:", filename)


def main():
    x_train, y_train, x_test, y_test = load_data()
    filters, convolution_bias, linear_weights, linear_bias = initialize_parameters()

    print("training images:", x_train.shape)
    train(
        x_train,
        y_train,
        filters,
        convolution_bias,
        linear_weights,
        linear_bias,
    )

    evaluate(
        x_test,
        y_test,
        filters,
        convolution_bias,
        linear_weights,
        linear_bias,
    )

    save_parameters(
        PARAMETER_FILE,
        filters,
        convolution_bias,
        linear_weights,
        linear_bias,
    )


if __name__ == "__main__":
    main()
