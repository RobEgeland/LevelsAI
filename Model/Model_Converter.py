import tensorflow as tf
import tf2onnx
import onnx

# Print versions for debugging
print(f"TensorFlow version: {tf.__version__}")
print(f"tf2onnx version: {tf2onnx.__version__}")

# Load the model
model = tf.keras.models.load_model(r'C:\Users\rober\Desktop\LevelsAI\LevelsAI\Model\eq_model.keras')

# Get the input shape from the model
input_shape = model.input_shape
if input_shape[0] is None:
    # Remove batch dimension and replace None with 1
    input_shape = (1,) + input_shape[1:]

# Create a function that represents your model
@tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32, name='input')])
def predict_func(x):
    return model(x)

# Convert the model using from_function
onnx_model, _ = tf2onnx.convert.from_function(
    predict_func,
    input_signature=[tf.TensorSpec(input_shape, tf.float32, name='input')],
    opset=13,
    output_path="model.onnx"
)

print("Conversion completed successfully!")
