import tensorflow as tf
import argparse

def read_a_tfrecord(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and print contents of a TFRecord file.")
    parser.add_argument("file_path", type=str, help="Path to the TFRecord file")
    args = parser.parse_args()

    read_a_tfrecord(args.file_path)