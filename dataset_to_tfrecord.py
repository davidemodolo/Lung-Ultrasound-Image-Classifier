import tensorflow as tf
import os
from PIL import Image
from tqdm import tqdm

# the result file is too large

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecords(png_folder_path, output_file_path):
    with tf.io.TFRecordWriter(output_file_path) as writer:
        for filename in tqdm(os.listdir(png_folder_path)):
            if filename.endswith('.png'):
                png_path = os.path.join(png_folder_path, filename)
                with Image.open(png_path) as image:
                    image_bytes = image.tobytes()
                    parts = os.path.splitext(filename)[0].split('_')
                    patient_id = parts[0]
                    exam_id = parts[1]
                    spot_id = parts[2]
                    frame_id = parts[3]
                    score = parts[4]
                    feature = {
                        'patient_id': _bytes_feature(patient_id.encode('utf-8')),
                        'exam_id': _bytes_feature(exam_id.encode('utf-8')),
                        'spot_id': _bytes_feature(spot_id.encode('utf-8')),
                        'frame_id': _bytes_feature(frame_id.encode('utf-8')),
                        'score': _bytes_feature(score.encode('utf-8')),
                        'image': _bytes_feature(image_bytes),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

# Example usage:
png_folder_path = 'images_png'
output_file_path = 'output.tfrecords'
convert_to_tfrecords(png_folder_path, output_file_path)
