import tensorflow as tf
import os

def is_image_valid(file_path):
    try:
        img_raw = tf.io.read_file(file_path)
        tf.image.decode_jpeg(img_raw, channels=3)  # Try to decode as JPEG
        return True
    except Exception as e:
        print(f"Corrupt: {file_path}")
        return False

def clean_directory(folder_path):
    removed = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            if not is_image_valid(path):
                os.remove(path)
                removed += 1
    print(f"\n🧹 Removed {removed} corrupted images.")

clean_directory("dataset")

