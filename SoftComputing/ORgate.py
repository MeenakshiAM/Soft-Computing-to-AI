import numpy as np

#this calculate the stepfn 
"""
if the input x is >= 0 then return 1 else return 0.
How it works:
If the input x is greater than or equal to zero, the function returns 1. This simulates an activation threshold where the perceptron "fires" and gives an output of 1.
If x is less than zero, the function returns 0. This means the perceptron doesn't "fire" and gives an output of 0.
"""
def step_function(x):
    return 1 if x >= 0 else 0

#perceptron fn
"""
np.dot(inputs, weights): This computes the dot product of the inputs and their corresponding weights. The dot product is the sum of the products of corresponding elements:

+ bias: The bias is added to this weighted sum to help shift the decision boundary of the perceptron.

step_function(summation): The weighted sum is passed through the step function to decide the output of the perceptron (either 1 or 0).
"""
def perceptron(inputs, weights, bias):
    summation = np.dot(inputs, weights) + bias
    return step_function(summation)

and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 possible inputs
and_output = np.array([0, 0, 0, 1])  # AND gate outputs

# OR Gate: Inputs and expected output
or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 possible inputs
or_output = np.array([0, 1, 1, 1])

# Define weights and bias for AND gate (initial random values)
weights_and = np.array([1, 1])  # Example: both weights set to 1
bias_and = -1.5  # This is adjusted based on the threshold for AND gate

# Define weights and bias for OR gate (initial random values)
weights_or = np.array([1, 1])  # Example: both weights set to 1
bias_or = -0.5  # This is adjusted based on the threshold for OR gate


# Test AND Gate
print("AND Gate Output:")
for i in range(4):
    result = perceptron(and_inputs[i], weights_and, bias_and)
    print(f"Input: {and_inputs[i]}, Predicted: {result}, Expected: {and_output[i]}")

# Test OR Gate
print("\nOR Gate Output:")
for i in range(4):
    result = perceptron(or_inputs[i], weights_or, bias_or)
    print(f"Input: {or_inputs[i]}, Predicted: {result}, Expected: {or_output[i]}")

"""
PS E:\New folder\Soft-Computing-to-AI\SoftComputing> python ORgate.py 
AND Gate Output:
Input: [0 0], Predicted: 0, Expected: 0
Input: [0 1], Predicted: 0, Expected: 0
Input: [1 0], Predicted: 0, Expected: 0
Input: [1 1], Predicted: 1, Expected: 1

OR Gate Output:
Input: [0 0], Predicted: 0, Expected: 0
Input: [0 1], Predicted: 1, Expected: 1
Input: [1 0], Predicted: 1, Expected: 1
Input: [1 1], Predicted: 1, Expected: 1
"""