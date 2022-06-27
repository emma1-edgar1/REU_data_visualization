"""
	TensorBoardExample.py
	Created by Adam Kohl
	4.8.2020

	Start TensorBoard through command line
	tensorboard --logdir logs/fit
"""
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds


def create_model():
	return tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dense(10, activation='softmax')
	])


def normalize_img(image, label):
	# Normalizes images: `uint8` -> `float32`
	return tf.cast(image, tf.float32) / 255., label


# ======================================================================================================================
# ========================================== CREATE THE INPUT PIPELINE =================================================
# ======================================================================================================================

""" Load the MNIST data file

shuffle_files: data is only stored in a single file,
			   but for larger datasets with multiple files on disk,
			   it's good practice to shuffle them when training
as_supervised: returns tuple (img, label)
			   instead of dict {'image: img, 'label': label} 
"""
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'],
										 shuffle_files=True, as_supervised=True, with_info=True)

""" Apply the following transformations

ds.map:		TFDS provide the images as tf.uint8, while the model expect tf.float32, so normalize images
ds.cache:	As the dataset fit in memory, cache before shuffling for better performance.
			Note: Random transformations should be applied after caching
ds.shuffle: For true randomness, set the shuffle buffer to the full dataset size.
			Note: For bigger datasets which do not fit in memory, a standard value is 1000 if your system allows it.
ds.batch: 	Batch after shuffling to get unique batches at each epoch.
ds.prefetch: Good practice to end the pipeline by prefetching for performances.
"""
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


""" Build the evaluation pipeline

	Testing pipeline is similar to the training pipeline, with small differences:
	No ds.shuffle() call
	Caching is done after batching (as batches can be the same between epoch)
"""
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# ======================================================================================================================
# ============================================== TENSORBOARD SETUP  ====================================================
# ======================================================================================================================
""" Establish the log configuration

Logs are placed in timestamped subdirectory for easy selection of training runs
Callback ensures that logs are created and stored
histogram_freq=1:	enables histogram computation every epoch
"""
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# ======================================================================================================================
# ============================================ MACHINE LEARNING MODEL ==================================================
# ======================================================================================================================
# Create the model
model = create_model()

# Configure the model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and validate the model
model.fit(ds_train, epochs=5, validation_data=ds_test, callbacks=[tensorboard_callback])
