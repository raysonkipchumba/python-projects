import math
import random
import sys

# --- Activation Functions ---
def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def d_leaky_relu(x):
    return 1.0 if x > 0 else 0.01

# --- Normalization Engine ---
class Normalizer:
    def __init__(self):
        self.in_min = []
        self.in_max = []
        self.out_min = []
        self.out_max = []
        self.initialized = False

    def init_bounds(self, num_in, num_out):
        self.in_min = [float('inf')] * num_in
        self.in_max = [float('-inf')] * num_in
        self.out_min = [float('inf')] * num_out
        self.out_max = [float('-inf')] * num_out
        self.initialized = True

    def update(self, inputs, expected=None):
        for i in range(len(inputs)):
            if inputs[i] < self.in_min[i]: self.in_min[i] = inputs[i]
            if inputs[i] > self.in_max[i]: self.in_max[i] = inputs[i]
        if expected:
            for i in range(len(expected)):
                if expected[i] < self.out_min[i]: self.out_min[i] = expected[i]
                if expected[i] > self.out_max[i]: self.out_max[i] = expected[i]

    def norm_inputs(self, inputs):
        res = []
        for i in range(len(inputs)):
            diff = self.in_max[i] - self.in_min[i]
            res.append(0.5 if diff == 0 else (inputs[i] - self.in_min[i]) / diff)
        return res

    def norm_expected(self, expected):
        res = []
        for i in range(len(expected)):
            diff = self.out_max[i] - self.out_min[i]
            res.append(0.5 if diff == 0 else (expected[i] - self.out_min[i]) / diff)
        return res

    def denorm_output(self, output):
        res = []
        for i in range(len(output)):
            diff = self.out_max[i] - self.out_min[i]
            res.append(self.out_min[i] if diff == 0 else (output[i] * diff) + self.out_min[i])
        return res


# --- Neural Network ---
class DeepMLP:
    def __init__(self, input_size, hidden_size=8, output_size=1, learning_rate=0.01):
        self.learning_rate = learning_rate
        # Strictly 10 hidden layers + input + output
        self.layer_sizes = [input_size] + [hidden_size] * 10 + [output_size]
        self.num_layers = len(self.layer_sizes)

        self.weights = []
        self.biases = []

        # Initialize network using He Uniform Initialization
        for i in range(self.num_layers - 1):
            limit = math.sqrt(6.0 / self.layer_sizes[i])
            w = [[random.uniform(-limit, limit) for _ in range(self.layer_sizes[i])] 
                 for _ in range(self.layer_sizes[i+1])]
            b = [0.0 for _ in range(self.layer_sizes[i+1])]
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        self.z_values = []
        self.activations = [inputs]
        a = inputs
        
        for i in range(self.num_layers - 1):
            z_layer = []
            a_layer = []
            for j in range(self.layer_sizes[i+1]):
                z = self.biases[i][j]
                for k in range(self.layer_sizes[i]):
                    z += self.weights[i][j][k] * a[k]
                z_layer.append(z)
                
                # Output layer uses Linear (no activation), hidden uses Leaky ReLU
                if i == self.num_layers - 2:
                    a_layer.append(z)
                else:
                    a_layer.append(leaky_relu(z))
                    
            self.z_values.append(z_layer)
            self.activations.append(a_layer)
            a = a_layer
        return a

    def backward(self, expected):
        deltas = [None] * (self.num_layers - 1)

        # 1. Output layer error (Derivative of Linear is 1)
        output_delta = []
        for j in range(len(expected)):
            error = self.activations[-1][j] - expected[j]
            error = max(min(error, 5.0), -5.0)  # Gradient clipping
            output_delta.append(error * 1.0)
        deltas[-1] = output_delta

        # 2. Backpropagate through 10 hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            layer_delta = []
            for j in range(self.layer_sizes[i+1]):
                err = 0.0
                for k in range(self.layer_sizes[i+2]):
                    err += deltas[i+1][k] * self.weights[i+1][k][j]
                layer_delta.append(err * d_leaky_relu(self.z_values[i][j]))
            deltas[i] = layer_delta

        # 3. Update Weights & Biases
        for i in range(self.num_layers - 1):
            for j in range(self.layer_sizes[i+1]):
                self.biases[i][j] -= self.learning_rate * deltas[i][j]
                for k in range(self.layer_sizes[i]):
                    self.weights[i][j][k] -= self.learning_rate * deltas[i][j] * self.activations[i][k]


# --- Interactive CLI Loop ---
def main():
    print("=== 10-Hidden-Layer Deep MLP (Pure Python) ===")
    
    try:
        num_inputs = int(input("How many inputs will the network take? (e.g., 2): "))
        num_outputs = int(input("How many outputs will it produce? (e.g., 1): "))
    except ValueError:
        print("Please enter integers.")
        sys.exit(1)

    nn = DeepMLP(input_size=num_inputs, hidden_size=8, output_size=num_outputs)
    norm = Normalizer()
    norm.init_bounds(num_inputs, num_outputs)
    
    dataset = [] # Stores raw data: tuples of (raw_inputs, raw_expected)

    print("\nNetwork initialized with strictly 10 hidden layers.")
    print("Commands:")
    print(f"  train <{num_inputs} inputs> : <{num_outputs} expected>  (e.g., 'train 1.5, 2.0 : 3.5')")
    print(f"  guess <{num_inputs} inputs>                      (e.g., 'guess 1.5, 2.0')")
    print("  stop")

    while True:
        try:
            cmd_raw = input("\n> ").strip().lower()
            if not cmd_raw: continue
            
            if cmd_raw == 'stop':
                print("Exiting...")
                break
                
            if cmd_raw.startswith('train'):
                # Parsing "train x, y : z"
                data_str = cmd_raw.replace('train', '').strip()
                in_str, out_str = data_str.split(':')
                
                raw_in = [float(x.strip()) for x in in_str.split(',')]
                raw_out = [float(x.strip()) for x in out_str.split(',')]
                
                if len(raw_in) != num_inputs or len(raw_out) != num_outputs:
                    print(f"Error: Expected {num_inputs} inputs and {num_outputs} outputs.")
                    continue
                    
                dataset.append((raw_in, raw_out))
                norm.update(raw_in, raw_out)
                
                # Train until it learns well
                print(f"Dataset size: {len(dataset)}. Training until convergence...")
                max_epochs = 2000
                target_loss = 0.001
                
                for epoch in range(max_epochs):
                    total_loss = 0
                    # Train on full collected dataset
                    for x, y in dataset:
                        norm_x = norm.norm_inputs(x)
                        norm_y = norm.norm_expected(y)
                        
                        out = nn.forward(norm_x)
                        nn.backward(norm_y)
                        
                        total_loss += sum((norm_y[k] - out[k])**2 for k in range(num_outputs))
                        
                    mse = total_loss / len(dataset)
                    if mse < target_loss:
                        print(f"Learned well! Hit MSE threshold {mse:.5f} at epoch {epoch}.")
                        break
                else:
                    print(f"Finished {max_epochs} epochs. Current MSE: {mse:.5f}")

            elif cmd_raw.startswith('guess'):
                # Parsing "guess x, y"
                data_str = cmd_raw.replace('guess', '').strip()
                raw_in = [float(x.strip()) for x in data_str.split(',')]
                
                if len(raw_in) != num_inputs:
                    print(f"Error: Expected {num_inputs} inputs.")
                    continue
                    
                if not dataset:
                    print("You need to train it on at least one data point first!")
                    continue
                
                # Dynamic normalization bounds check
                norm.update(raw_in)
                norm_x = norm.norm_inputs(raw_in)
                
                raw_pred = nn.forward(norm_x)

                final_guess = norm.denorm_output(raw_pred)
                
                print(f"Network Guesses: {[round(g, 4) for g in final_guess]}")
                
            else:
                print("Unknown command.")
                
        except Exception as e:
            print(f"Error parsing input: {e}. Check your formatting.")

if __name__ == "__main__":
    main()
