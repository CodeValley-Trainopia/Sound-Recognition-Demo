import numpy as np
import sounddevice as sd
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Whistle_Vs_Shaker_Vs_Clap.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Get target input shape (e.g., [1, 15600])
target_shape = input_details[0]['shape']
target_samples = target_shape[1]

# Load labels
with open('labels.txt', 'r') as f:
    labels = []
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            labels.append(parts[1])
        else:
            labels.append(parts[0])

# Audio settings
sample_rate = 44032

def capture_and_classify():
    while True:
        try:
            # Record target_samples at the correct sample rate
            recording = sd.rec(target_samples, samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            audio_data = recording.flatten()

            # Preprocess: Pad/truncate to match exactly target_samples
            if len(audio_data) < target_samples:
                padding = target_samples - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            else:
                audio_data = audio_data[:target_samples]

            # Normalize audio to range [-1.0, 1.0]
            max_val = np.max(np.abs(audio_data)) + 1e-6  # Avoid division by zero
            audio_data = audio_data / max_val

            # Reshape for model input
            input_data = np.expand_dims(audio_data, axis=0).astype(np.float32)

            # Debugging shapes

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get top class
            predicted_index = np.argmax(output_data)
            confidence = output_data[predicted_index]
            label = labels[predicted_index]

            print(f"🔎 Detected: {label} (Confidence: {confidence:.2f})")

        except KeyboardInterrupt:
            print("\n👋 Exiting real-time classification...")
            break
        except Exception as e:
            print(f"❌ Error during classification: {e}")

if __name__ == "__main__":
    print("\n🔊 Real-time Streaming Sound Classification started! Ctrl+C to stop.")
    capture_and_classify()
