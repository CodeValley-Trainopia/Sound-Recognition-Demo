import numpy as np
import sounddevice as sd
import tensorflow as tf

class SoundClassifier:
    def __init__(self, model_path="Whistle_Vs_Shaker_Vs_Clap.tflite", labels_path="labels.txt", sample_rate=44032):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Determine target input sample length
        self.target_shape = self.input_details[0]['shape']
        self.target_samples = self.target_shape[1]
        self.sample_rate = sample_rate

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = []
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    self.labels.append(parts[1])
                else:
                    self.labels.append(parts[0])


    def capture_audio(self):
        recording = sd.rec(self.target_samples, samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_data = recording.flatten()

        # Pad or trim
        if len(audio_data) < self.target_samples:
            padding = self.target_samples - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        else:
            audio_data = audio_data[:self.target_samples]

        # Normalize between -1 and 1
        max_val = np.max(np.abs(audio_data)) + 1e-6
        audio_data = audio_data / max_val

        # Reshape for model
        input_data = np.expand_dims(audio_data, axis=0).astype(np.float32)
        return input_data

    def predict(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        predicted_index = np.argmax(output_data)
        confidence = output_data[predicted_index]
        label = self.labels[predicted_index]
        return label, confidence

    def run_real_time_classification(self):
        print("\n🔊 Real-time Sound Classification started! Ctrl+C to stop.")
        try:
            while True:
                input_data = self.capture_audio()
                label, confidence = self.predict(input_data)
                print(f"🔎 Detected: {label} (Confidence: {confidence:.2f})")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        except Exception as e:
            print(f"❌ Error: {e}")
