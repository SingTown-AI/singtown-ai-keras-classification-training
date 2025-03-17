import keras
from keras import layers
from tensorflow import data as tf_data
import argparse

parser = argparse.ArgumentParser(description="Train Classification with Keras")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--weight", type=str, default="imagenet", help="model weight")
parser.add_argument("--alpha", type=float, default=1.0, help="model alpha")
parser.add_argument("--imgw", type=int, help="input image width")
parser.add_argument("--imgh", type=int, help="input image height")
parser.add_argument("--labels", nargs="+", type=str, help="A list of strings")
parser.add_argument("--epochs", type=int, help="task token")
parser.add_argument("--learning_rate", type=float, default=0.001, help="task token")
args = parser.parse_args()

MODEL_NAME = args.model
WEIGHT = args.weight
ALPHA = args.alpha
IMG_W = args.imgw
IMG_H = args.imgh
LABELS = args.labels
EPOCHS = args.epochs
BATCH_SIZE = 32
LEARNING_RATE = args.learning_rate


data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def load_training_data():
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "resource",
        validation_split=0.2,
        subset="both",
        seed=1337,
        shuffle=True,
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=LABELS,
    )
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img) / 127.5 - 1, label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda img, label: (img / 127.5 - 1, label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    return train_ds, val_ds


def build_model():
    input_shape = (IMG_H, IMG_W, 3)
    inputs = layers.Input(shape=input_shape)

    if MODEL_NAME == "MobileNetV1":
        model = keras.applications.MobileNet(
            alpha=ALPHA,
            include_top=False,
            input_shape=input_shape,
            input_tensor=inputs,
            weights=WEIGHT,
        )
    elif MODEL_NAME == "MobileNetV2":
        model = keras.applications.MobileNetV2(
            alpha=ALPHA,
            include_top=False,
            input_shape=input_shape,
            input_tensor=inputs,
            weights=WEIGHT,
        )
    else:
        raise RuntimeError("model not available:" + MODEL_NAME)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.1
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(len(LABELS), activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="MobileNet")
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def callbacks():
    earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    saveBest = keras.callbacks.ModelCheckpoint(
        "best.keras",
        monitor="val_loss",
        save_best_only=True,
    )
    logger = keras.callbacks.CSVLogger("metrics.csv")
    return [earlyStopping, saveBest, logger]


train_ds, val_ds = load_training_data()

model = build_model()
hist = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks(),
    verbose=2,
)
