import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

"""
Loading the images
"""

train_dir = "train"


train_dataset = tf.keras.utils.image_dataset_from_directory(directory = train_dir,
                                                            label_mode='int',
                                                            batch_size=8,
                                                            image_size=(256,256),
                                                            validation_split=0.2,
                                                            color_mode="grayscale",
                                                            seed=42,
                                                            subset="training")

validation_dataset = tf.keras.utils.image_dataset_from_directory(directory = train_dir,
                                                                 label_mode="int",
                                                                 batch_size=8,
                                                                 image_size=(256,256),
                                                                 validation_split=0.2,
                                                                 color_mode="grayscale",
                                                                 seed=42,
                                                                 subset="validation")



"""
Creating the model
"""

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(256,256,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

"""
Fit the model
"""

history = model.fit(train_dataset, epochs=3, validation_data=validation_dataset)

"""
Evaluate the model
"""

model.evaluate(validation_dataset)

"""
Plot the history
"""

pd.DataFrame(history.history).plot(figsize=(10,10))
plt.show()

"""
Save the model
"""

model.save("model")


