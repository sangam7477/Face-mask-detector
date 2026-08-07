import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_cnn_model(input_shape=(128, 128, 3), lr=1e-3):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_model(input_shape=(224, 224, 3), lr=1e-4, pretrained=False):
    base = MobileNetV2(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_tensor=Input(shape=input_shape),
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(arch="cnn", input_shape=(128, 128, 3), lr=1e-3, pretrained=False):
    if arch == "mobilenetv2":
        return build_mobilenet_model(input_shape=input_shape, lr=lr, pretrained=pretrained)
    return build_cnn_model(input_shape=input_shape, lr=lr)


def compute_class_weights(generator):
    counts = np.bincount(generator.classes)
    total = counts.sum()
    class_weight = {}
    for idx, count in enumerate(counts):
        if count > 0:
            class_weight[idx] = total / (len(counts) * count)
    return class_weight


def inspect_dataset(dataset_dir):
    required_classes = ["with_mask", "without_mask"]
    counts = {}

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_dir}. "
            "Expected subfolders: with_mask/ and without_mask/."
        )

    for class_name in required_classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing required class directory: {class_dir}"
            )

        image_count = 0
        for root, _, files in os.walk(class_dir):
            for file_name in files:
                if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS:
                    image_count += 1
        counts[class_name] = image_count

    if any(count == 0 for count in counts.values()):
        raise ValueError(
            "Dataset is empty. Add images to both "
            "`dataset/with_mask` and `dataset/without_mask` before training."
        )

    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face mask detector model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset folder (with_mask/without_mask)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mask_detector_model.h5",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="models/labels.json",
        help="Path to save label ordering",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/model_config.json",
        help="Path to save model metadata such as labels and input size",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["cnn", "mobilenetv2"],
        default="cnn",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet weights for MobileNetV2. Requires internet on first run.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=128, help="Input image size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_counts = inspect_dataset(args.dataset)
    print(f"Dataset summary: {dataset_counts}")

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_aug.flow_from_directory(
        args.dataset,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_aug.flow_from_directory(
        args.dataset,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    if train_gen.samples == 0 or val_gen.samples == 0:
        raise ValueError(
            "Not enough images to create train/validation splits. "
            "Add more images to each class and try again."
        )

    model = build_model(
        arch=args.arch,
        input_shape=(args.image_size, args.image_size, 3),
        lr=args.lr,
        pretrained=args.pretrained,
    )

    class_weight = compute_class_weights(train_gen)
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output, include_optimizer=False)

    label_list = [None] * len(train_gen.class_indices)
    for name, idx in train_gen.class_indices.items():
        label_list[int(idx)] = name
    os.makedirs(os.path.dirname(args.labels) or ".", exist_ok=True)
    with open(args.labels, "w", encoding="utf-8") as f:
        json.dump(label_list, f, indent=2)

    config = {
        "architecture": args.arch,
        "image_size": [args.image_size, args.image_size],
        "labels": label_list,
        "dataset_counts": dataset_counts,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    os.makedirs(os.path.dirname(args.config) or ".", exist_ok=True)
    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {args.output}")
    print(f"Labels saved to: {args.labels}")
    print(f"Config saved to: {args.config}")


if __name__ == "__main__":
    main()
