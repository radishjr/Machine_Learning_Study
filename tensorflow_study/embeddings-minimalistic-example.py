import tensorflow as tf
import numpy as np

from PIL import Image

from tensorflow.contrib.tensorboard.plugins import projector
import os

current_path = os.path.dirname(os.path.abspath(__file__)) + "/"
path = current_path + "tmp/logs"

def load_data():
    features = [
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    features = np.array(features)
    labels = [
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0
    ]
    labels = np.array(labels)
    return (features, labels)


def create_tensor(features):
    embedding_variable = tf.Variable(features, name='embedding')
    return embedding_variable


def define_embedding(embedding_variable, dimensions):
    summary_writer = tf.summary.FileWriter(path)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_variable.name
    embedding.metadata_path = path + '/metadata.tsv'
    embedding.sprite.image_path = path + '/sprites.png'
    embedding.sprite.single_image_dim.extend(dimensions)
    projector.visualize_embeddings(summary_writer, config)
    return embedding


def run_tensorflow():
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    return session


def save_checkpoint(session):
    saver = tf.train.Saver()
    saver.save(session, path + '/model.ckpt', 0)


def create_sprites(dimensions):
    sprites = [None] * 2
    sprites[0] = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    sprites[1] = [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    sprites[0] = Image.fromarray(np.uint8(sprites[0]) * 0xFF)
    sprites[1] = Image.fromarray(np.uint8(sprites[1]) * 0xFF)
    sprites[0] = sprites[0].resize(dimensions, Image.NEAREST)
    sprites[1] = sprites[1].resize(dimensions, Image.NEAREST)
    sprites[0].save(path + '/sprite0.png')
    sprites[1].save(path + '/sprite1.png')
    return sprites


def merge_sprites(labels, embedding, single, sprites):
    count = labels.shape[0]
    size = int(np.ceil(np.sqrt(count)))
    merged = Image.new('1', (size * single, size * single))
    for i, label in enumerate(labels):
        there = ((i % size) * single, (i // size) * single)
        merged.paste(sprites[label], there)
    merged.save(embedding.sprite.image_path)


def create_metadata(labels, embedding):
    with open(embedding.metadata_path, 'w') as handle:
        for label in labels:
            handle.write('{}\n'.format(label))


def main():
    features, labels = load_data()
    embedding_variable = create_tensor(features)
    single = 100
    dimensions = [100, 100]
    embedding = define_embedding(embedding_variable, dimensions)
    session = run_tensorflow()
    save_checkpoint(session)
    sprites = create_sprites(dimensions)
    merge_sprites(labels, embedding, single, sprites)
    create_metadata(labels, embedding)


if __name__ == "__main__":
    main()