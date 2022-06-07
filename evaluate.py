import random
import tensorflow as tf
import os
import matplotlib.pyplot as plt

test_dir = "test/"

model = tf.keras.models.load_model("model")

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

label_names = ["NORMAL", "PNEUMONIA"]


def prep_file_and_prediction(filename, label, model):

    img = tf.io.read_file(filename)

    img = tf.image.decode_image(img)

    img = tf.image.resize(img, size=(256,256))

    prediction = model.predict(tf.expand_dims(img, axis=0))

    prediction = int(tf.round(prediction))

    if label_names[prediction] == label:
        c="green"
    else:
        c="red"

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title(f"Prediction {label_names[prediction]}", color=c)
    plt.axis(False)
    plt.show()


for i in range(9):

    random_index = random.choice(os.listdir(test_dir))
    random_path = test_dir + random_index  + "/"
    random_img = random.choice(os.listdir(random_path))

    image = random_path + random_img

    prep_file_and_prediction(image, random_index, model)






