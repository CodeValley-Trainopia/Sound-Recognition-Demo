from sound_classifier import SoundClassifier

if __name__ == "__main__":
    classifier = SoundClassifier(
        model_path="Whistle_Vs_Shaker_Vs_Clap.tflite",
        labels_path="labels.txt"
    )
    classifier.run_real_time_classification()
