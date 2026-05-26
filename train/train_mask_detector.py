from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from imutils import paths

import numpy as np
import cv2
import os

INIT_LR = 1e-4
EPOCHS = 10
BS = 32

dataset = "../Facemaskdataset/train"

print("[INFO] loading images...")

data = []
labels = []

imagePaths = list(paths.list_images(dataset))
print("So luong anh:", len(imagePaths))

for imagePath in imagePaths:

    label = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(7, 7))(headModel)

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(128, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(learning_rate=INIT_LR)

model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

print("[INFO] training head...")

model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=max(1, len(trainX) // BS),
    validation_data=(testX, testY),
    validation_steps=max(1, len(testX) // BS),
    epochs=EPOCHS
)

print("[INFO] saving model...")

model.save("../models/mask_detector.keras")