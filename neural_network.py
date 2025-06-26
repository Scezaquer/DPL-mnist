import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions


def sigmoid(x):
    """Numerically stable sigmoid function."""
    x = np.clip(x, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    sig = sigmoid(x)
    return sig * (1.0 - sig)


def softmax(x):
    """Numerically stable softmax function."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Weight initialization


def initialize_weights(seed, dims):
    """Initialize weights with random values between -0.5 and 0.5."""
    np.random.seed(seed)
    weights = []
    for i in range(len(dims) - 1):
        # DPL version adds bias implicitly, we add it to the input size
        w = np.random.rand(dims[i] + 1, dims[i+1]) - 0.5
        weights.append(w)
    return weights

# Forward pass


def predict(input_sample, weights):
    """Forward pass through the network to get activations."""
    activations = [np.append(input_sample, 1)]  # Add bias term to input

    for i, w in enumerate(weights):
        net_input = np.dot(activations[-1], w)

        # Apply sigmoid to all but the last layer
        if i < len(weights) - 1:
            activation = sigmoid(net_input)
        else:
            # No activation on the final output before loss calculation in
            # this architecture
            activation = net_input

        # Add bias term for the next layer, except for the output layer
        if i < len(weights) - 1:
            activation = np.append(activation, 1)

        activations.append(activation)

    return activations

# Training function


def fit(X_train, y_train, weights, dims, lr, epochs, batch_size, lr_decay):
    """Train the neural network using backpropagation with minibatches."""
    print("Starting training...")

    current_lr = lr

    for epoch in range(epochs):
        correct = 0
        total_loss = 0

        # Shuffle training data
        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]

            for x, y_true in zip(batch_X, batch_y):
                # Forward pass
                activations = predict(x, weights)

                # Output layer activation (apply sigmoid for probability)
                output = sigmoid(activations[-1])
                activations[-1] = output  # Store the final probability

                # --- Loss Calculation (Cross-Entropy) ---
                # Create one-hot encoded target vector
                target = np.zeros(dims[-1])
                target[y_true] = 1

                # Clip predictions to avoid log(0)
                output_clipped = np.clip(output, 1e-12, 1. - 1e-12)
                loss = -np.sum(target * np.log(output_clipped) +
                               (1 - target) * np.log(1 - output_clipped))
                total_loss += loss

                # --- Tally correct predictions ---
                predicted_class = np.argmax(output)
                if predicted_class == y_true:
                    correct += 1

                # --- Backward Pass (Backpropagation) ---
                # Error at the output layer
                error = output - target

                # Propagate error backwards
                for j in range(len(weights) - 1, -1, -1):
                    # Get activations for the current layer (input to the
                    # weights being updated)
                    # The activation list includes the initial input, so it's
                    # one longer than the weights list
                    prev_activations = activations[j]

                    # Calculate gradient for the current layer's weights
                    # Reshape prev_activations to be a column vector and error
                    # to be a row vector
                    gradient = np.outer(prev_activations, error)

                    # Update weights
                    weights[j] -= current_lr * gradient

                    # Calculate error for the next (previous) layer if it's
                    # not the input layer
                    if j > 0:
                        # Propagate error to the previous layer
                        # Exclude bias from weight matrix during error
                        # propagation
                        error_propagated = np.dot(weights[j][:-1, :], error)

                        # Get the derivative of the activation (before bias
                        # was added)
                        # The activation at j is the input to layer j, its
                        # derivative is needed
                        derivative = sigmoid_derivative(activations[j][:-1])
                        error = error_propagated * derivative

        # --- End of Epoch ---
        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Loss = {avg_loss:.4f}, "
            f"Accuracy = {accuracy * 100:.2f}%, "
            f"Learning Rate = {current_lr:.6f}"
        )

        # Decay learning rate
        current_lr *= lr_decay

    return weights

# Testing function


def test(X_test, y_test, weights):
    """Test the neural network and return accuracy."""
    correct = 0
    for x, y_true in zip(X_test, y_test):
        # Forward pass
        activations = predict(x, weights)
        output = sigmoid(activations[-1])  # Final activation

        # Get predicted class
        predicted_class = np.argmax(output)

        # Check if prediction is correct
        if predicted_class == y_true:
            correct += 1

    accuracy = correct / len(X_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct: {correct}, Incorrect: {len(X_test) - correct}")
    return accuracy


# Main execution block
if __name__ == "__main__":
    # --- Configuration ---
    # Network architecture: 64 inputs -> 32 hidden -> 32 hidden -> 10 outputs
    # The DPL file had 65 inputs, implying a bias unit. We handle bias by
    # adding it to activations.
    DIMS = [64, 32, 32, 10]
    LR = 0.01          # Learning rate
    EPOCHS = 35        # Number of training epochs
    BATCH_SIZE = 32    # Size of each training batch
    LR_DECAY = 0.99    # Learning rate decay factor
    SEED = 42          # Random seed for reproducibility

    # --- Load and Prepare Data ---
    print("Loading and preparing MNIST dataset...")
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(
        f"Dataset loaded: {len(X_train)} training samples, {len(X_test)}"
        "testing samples.")

    # --- Initialize and Train ---
    # Initialize weights randomly
    weights = initialize_weights(SEED, DIMS)

    # Train the model
    trained_weights = fit(X_train, y_train, weights, DIMS,
                          LR, EPOCHS, BATCH_SIZE, LR_DECAY)

    # --- Test the Model ---
    test(X_test, y_test, trained_weights)
