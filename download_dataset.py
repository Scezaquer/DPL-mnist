from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("model_weights.dpl", "w") as file:
    file.write("""// 65 (8*8 + 1) input neurons
// 32 hidden neurons in layer 1
// 32 hidden neurons in layer 2
// 10 output neurons (for digits 0-9)

""")
    file.write(f"let input_layer: array[float] = {[0.0] * 65};\n")
    file.write(f"let activation1: array[float] = {[0.0] * 32};\n")
    file.write(f"let activation2: array[float] = {[0.0] * 32};\n")
    file.write(f"let output_layer: array[float] = {[0.0] * 10};\n")

    file.write("let activations: array[array[float]] = [input_layer, activation1, activation2, output_layer];\n\n\n")

    file.write(f"let weights1[32, 65]: array[float] = {[0.0] * 65 * 32};\n")
    file.write(f"let weights2[32, 32]: array[float] = {[0.0] * 32 * 32};\n")
    file.write(f"let weights3[10, 32]: array[float] = {[0.0] * 32 * 10};\n")

    file.write("let weights: array[array[float]] = [weights1, weights2, weights3];\n\n")
    file.write("let dims: array[int] = [65, 32, 32, 10];\n\n")

    file.write(f"let grad1[32, 65]: array[float] = {[0.0] * 65 * 32};\n")
    file.write(f"let grad2[32, 32]: array[float] = {[0.0] * 32 * 32};\n")
    file.write(f"let grad3[10, 32]: array[float] = {[0.0] * 32 * 10};\n")

    file.write("let grad: array[array[float]] = [grad1, grad2, grad3];\n\n")

    file.write(f"let net_inputs1[32]: array[float] = {[0.0] * 32};\n")
    file.write(f"let net_inputs2[32]: array[float] = {[0.0] * 32};\n")
    file.write(f"let net_inputs3[10]: array[float] = {[0.0] * 10};\n")
    file.write("let net_inputs: array[array[float]] = [net_inputs1, net_inputs2, net_inputs3];\n\n")

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=True
)

# X_train, y_train = X_train[:10], y_train[:10]  # Limit to 10 samples for testing

file_content = ""

for (index, i) in enumerate(X_train):
    file_content += f"let img{index}: array[float] = {[1.0] + [float(x) for x in i]};\n"
file_content += "let mnist_train: array[array[float]] = ["
for i in range(len(X_train)):
    file_content += f"img{i}, "
file_content = file_content[:-2]  # Remove the last comma
file_content += "];\n\n\n"

file_content += "let mnist_train_labels: array[int] = ["
for i in tqdm(y_train):
    file_content += f"{i}, "
file_content = file_content[:-2]  # Remove the last comma
file_content += "];\n"

with open("mnist_train.dpl", "w") as file:
    file.write(file_content)


file_content = ""

for (index, i) in enumerate(X_test):
    file_content += f"let test_img{index}: array[float] = {[1.0] + [float(x) for x in i]};\n"
file_content += "let mnist_test: array[array[float]] = ["
for i in range(len(X_test)):
    file_content += f"test_img{i}, "
file_content = file_content[:-2]  # Remove the last comma
file_content += "];\n\n\n"

file_content += "let mnist_test_labels: array[int] = ["
for i in tqdm(y_test):
    file_content += f"{i}, "
file_content = file_content[:-2]  # Remove the last comma
file_content += "];\n"

with open("mnist_test.dpl", "w") as file:
    file.write(file_content)