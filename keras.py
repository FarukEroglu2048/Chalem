import random

import numpy
import scipy

import tensorflow
import tensorflow_datasets

from tensorflow.python.keras import (backend, layers, models)
                                
word_length = 16
window_size = 8

embedding_size = 384

epoch_count = 2

train_batch_size = 256
validation_batch_size = 256

train_cache_size = 65536
validation_cache_size = 65536

pad_character = "ψ"

dictionary = pad_character + """QWERTYUIOPASDFGHJKLZXCVBNMqwertyuopasdfghjklizxcvbnm1234567890.,:;!?'`"^&#%@$+-*/=_~\\|()[]{}<> \r\n"""

dataset = tensorflow_datasets.load("squad")

dataset_numpy = tensorflow_datasets.as_numpy(dataset)

train_word_set = list()
validation_word_set = list()

for item in dataset_numpy["train"]:
    text = str(item["context"], encoding="ascii", errors="ignore")

    train_word_set.extend(text.split(" "))

for item in dataset_numpy["validation"]:
    text = str(item["context"], encoding="ascii", errors="ignore")

    validation_word_set.extend(text.split(" "))

train_data_size = len(train_word_set) * 2
validation_data_size = len(validation_word_set) * 2

print(train_data_size)
print(validation_data_size)

def model_generator(mode, batch_size, cache_size):
    dataset_numpy = tensorflow_datasets.as_numpy(dataset)

    while True:
        batch_samples = list()
        
        while(len(batch_samples) < cache_size):
            try:
                dataset_sample = next(dataset_numpy[mode])
            except StopIteration:
                dataset_numpy = tensorflow_datasets.as_numpy(dataset)
                dataset_sample = next(dataset_numpy[mode])

            text = str(dataset_sample["context"], encoding="ascii", errors="ignore")

            batch_samples.extend(text.split(" "))

        batch_word_set = set(batch_samples)

        batch_samples_input_1 = numpy.zeros((len(batch_samples) * 2, word_length, len(dictionary)), dtype=numpy.uint8)
        batch_samples_input_2 = numpy.zeros((len(batch_samples) * 2, window_size * 2, word_length, len(dictionary)), dtype=numpy.uint8)

        batch_targets_output_1 = numpy.zeros((len(batch_samples) * 2, 1), dtype=numpy.uint8)

        for sample_index, sample_word in enumerate(batch_samples):
            start_index = max(0, sample_index - window_size)
            stop_index = min(len(batch_samples) - 1, sample_index + window_size)

            if (sample_index + window_size) > stop_index:
                start_index -= window_size - (stop_index - sample_index)
            elif (sample_index - window_size) < start_index:
                stop_index += window_size - (sample_index - start_index)

            context_list = batch_samples[start_index:stop_index + 1]
            random_list = random.sample(batch_word_set.difference(context_list), window_size * 2)

            context_list.remove(sample_word)

            sample_word = sample_word.center(word_length, pad_character)
            sample_word = sample_word[:word_length]

            for char_index, char in enumerate(sample_word):
                batch_samples_input_1[sample_index * 2, char_index, dictionary.index(char)] = 1
                batch_samples_input_1[sample_index * 2 + 1, char_index, dictionary.index(char)] = 1

            for context_index, context_word in enumerate(context_list):
                context_word = context_word.center(word_length, pad_character)
                context_word = context_word[:word_length]

                for char_index, char in enumerate(context_word):
                    batch_samples_input_2[sample_index * 2, context_index, char_index, dictionary.index(char)] = 1

            for random_index, random_word in enumerate(random_list):
                random_word = random_word.center(word_length, pad_character)
                random_word = random_word[:word_length]

                for char_index, char in enumerate(random_word):
                    batch_samples_input_2[sample_index * 2 + 1, random_index, char_index, dictionary.index(char)] = 1
                
            batch_targets_output_1[sample_index * 2, :] = 1
            batch_targets_output_1[sample_index * 2 + 1, :] = 0

            if ((sample_index * 2 + 2) % batch_size) == 0:
                batch_stop_index = sample_index * 2 + 2
                batch_start_index = batch_stop_index - batch_size

                yield [batch_samples_input_1[batch_start_index:batch_stop_index, :, :], batch_samples_input_2[batch_start_index:batch_stop_index, :, :, :]], [batch_samples_input_1[batch_start_index:batch_stop_index, :, :], batch_targets_output_1[batch_start_index:batch_stop_index, :]]
            elif sample_index == (len(batch_samples) - 1):
                batch_start_index = batch_stop_index
                batch_stop_index = len(batch_samples) * 2

                yield [batch_samples_input_1[batch_start_index:batch_stop_index, :, :], batch_samples_input_2[batch_start_index:batch_stop_index, :, :, :]], [batch_samples_input_1[batch_start_index:batch_stop_index, :, :], batch_targets_output_1[batch_start_index:batch_stop_index, :]]

def cosine_proximity(vectors):
    normalized_1 = backend.l2_normalize(vectors[0], axis=-1)
    normalized_2 = backend.l2_normalize(vectors[1], axis=-1)

    return backend.math_ops.reduce_sum(normalized_1 * normalized_2, axis=-1)

def accuracy_ignore_padding(y_true, y_pred):
    prediction = backend.argmax(y_pred, axis=-1)
    target = backend.argmax(y_true, axis=-1)

    accuracy = backend.equal(backend.array_ops.boolean_mask(prediction, backend.not_equal(target, 0)), backend.array_ops.boolean_mask(target, backend.not_equal(target, 0)))

    return backend.mean(accuracy)

concatenate = layers.Concatenate()
flatten = layers.Flatten()

input_1 = layers.Input(shape=(word_length, len(dictionary)))

convolution_1 = layers.Conv1D(32, 2, padding="same", activation="tanh")
convolution_2 = layers.Conv1D(32, 3, padding="same", activation="tanh")
convolution_3 = layers.Conv1D(32, 5, padding="same", activation="tanh")
convolution_4 = layers.Conv1D(32, 7, padding="same", activation="tanh")

convolution_1_output = convolution_1(input_1)
convolution_2_output = convolution_2(input_1)
convolution_3_output = convolution_3(input_1)
convolution_4_output = convolution_4(input_1)

convolution_layer_1_output = concatenate([convolution_1_output, convolution_2_output, convolution_3_output, convolution_4_output])

convolution_5 = layers.Conv1D(32, 2, padding="same", activation="tanh")
convolution_6 = layers.Conv1D(32, 3, padding="same", activation="tanh")
convolution_7 = layers.Conv1D(32, 5, padding="same", activation="tanh")
convolution_8 = layers.Conv1D(32, 7, padding="same", activation="tanh")

convolution_5_output = convolution_5(convolution_layer_1_output)
convolution_6_output = convolution_6(convolution_layer_1_output)
convolution_7_output = convolution_7(convolution_layer_1_output)
convolution_8_output = convolution_8(convolution_layer_1_output)

convolution_layer_2_output = concatenate([convolution_5_output, convolution_6_output, convolution_7_output, convolution_8_output])

reshape_1 = layers.Reshape((1, word_length, 128))
reshape_1_output = reshape_1(convolution_layer_2_output)

depthwise_convolution_1 = layers.DepthwiseConv2D((1, word_length), depth_multiplier=8, activation="tanh")
depthwise_convolution_1_output = depthwise_convolution_1(reshape_1_output)

flatten_output_1 = flatten(depthwise_convolution_1_output)

dense_1 = layers.Dense(embedding_size, activation="tanh")
dense_1_output = dense_1(flatten_output_1)

encoder = models.Model(inputs=[input_1], outputs=[dense_1_output])

encoder.summary()

input_2 = layers.Input(shape=(embedding_size,))

dense_2 = layers.Dense(word_length * 32, activation="tanh")
dense_2_output = dense_2(input_2)

reshape_2 = layers.Reshape((word_length, 32))
reshape_2_output = reshape_2(dense_2_output)

convolution_9 = layers.Conv1D(32, 2, padding="same", activation="tanh")
convolution_10 = layers.Conv1D(32, 3, padding="same", activation="tanh")
convolution_11 = layers.Conv1D(32, 5, padding="same", activation="tanh")
convolution_12 = layers.Conv1D(32, 7, padding="same", activation="tanh")

convolution_9_output = convolution_9(reshape_2_output)
convolution_10_output = convolution_10(reshape_2_output)
convolution_11_output = convolution_11(reshape_2_output)
convolution_12_output = convolution_12(reshape_2_output)

convolution_layer_3_output = concatenate([convolution_9_output, convolution_10_output, convolution_11_output, convolution_12_output])

convolution_13 = layers.Conv1D(32, 2, padding="same", activation="tanh")
convolution_14 = layers.Conv1D(32, 3, padding="same", activation="tanh")
convolution_15 = layers.Conv1D(32, 5, padding="same", activation="tanh")
convolution_16 = layers.Conv1D(32, 7, padding="same", activation="tanh")

convolution_13_output = convolution_13(convolution_layer_3_output)
convolution_14_output = convolution_14(convolution_layer_3_output)
convolution_15_output = convolution_15(convolution_layer_3_output)
convolution_16_output = convolution_16(convolution_layer_3_output)

convolution_layer_4_output = concatenate([convolution_13_output, convolution_14_output, convolution_15_output, convolution_16_output])

convolution_17 = layers.Conv1D(len(dictionary), 3, padding="same", activation=None)
convolution_17_output = convolution_17(convolution_layer_4_output)

activation_1 = layers.Activation("softmax")
activation_1_output = activation_1(convolution_17_output)

decoder = models.Model(inputs=[input_2], outputs=[activation_1_output])

decoder.summary()

input_3 = layers.Input(shape=(word_length, len(dictionary)))
input_4 = layers.Input(shape=(window_size * 2, word_length, len(dictionary)))

encoder_output_1 = encoder(input_3)

repeat_vector_1 = layers.RepeatVector(window_size * 2)
repeat_vector_1_output = repeat_vector_1(encoder_output_1)

timedistributed_encoder_1 = layers.TimeDistributed(encoder)
timedistributed_encoder_1_output = timedistributed_encoder_1(input_4)

lambda_1 = layers.Lambda(cosine_proximity)
lambda_1_output = lambda_1([repeat_vector_1_output, timedistributed_encoder_1_output])

dense_3 = layers.Dense(1, activation="sigmoid")
dense_3_output = dense_3(lambda_1_output)

decoder_output_1 = decoder(encoder_output_1)

model = models.Model(inputs=[input_3, input_4], outputs=[decoder_output_1, dense_3_output])

model.compile(optimizer="rmsprop", loss=["categorical_crossentropy", "binary_crossentropy"], metrics={"model_1": accuracy_ignore_padding, "dense_2": "accuracy"})

model.summary()

for epoch in range(epoch_count):
    model.fit_generator(model_generator("train", train_batch_size, train_cache_size), steps_per_epoch=train_data_size // train_batch_size, validation_data=model_generator("validation", validation_batch_size, validation_cache_size), validation_steps=validation_data_size // validation_batch_size)

    encoder.save("Encoder-Model-" + str(epoch + 1) + ".HDF5")
    decoder.save("Decoder-Model-" + str(epoch + 1) + ".HDF5")
