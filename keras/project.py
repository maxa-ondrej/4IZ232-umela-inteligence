import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

df = pd.read_csv(
    "amazon_cells_labelled.txt", sep="\t", header=None, names=["text", "label"]
)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_end = int(0.7 * len(df))
val_end = int(0.85 * len(df))

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

X_train, y_train = train["text"], train["label"]
X_val, y_val = val["text"], val["label"]
X_test, y_test = test["text"], test["label"]


def create_datasets(model, max_tokens, max_len, batch_size, loss, optimizer):
    y_tr = y_train
    y_v = y_val
    y_te = y_test
    if loss == "hinge":
        y_tr = y_tr.map(lambda x: 1 if x == 1 else -1)
        y_v = y_v.map(lambda x: 1 if x == 1 else -1)
        y_te = y_te.map(lambda x: 1 if x == 1 else -1)
    vectorize = layers.TextVectorization(
        max_tokens=max_tokens, output_mode="int", output_sequence_length=max_len
    )

    vectorize.adapt(X_train.values)

    def make_ds(texts, labels):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        ds = ds.batch(batch_size).map(lambda x, y: (vectorize(x), y))
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train.values, y_tr.values)
    val_ds = make_ds(X_val.values, y_v.values)
    test_ds = make_ds(X_test.values, y_te.values)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.summary()
    return train_ds, val_ds, test_ds


def use_model(name, epochs, model, train_ds, val_ds, test_ds):
    folder = "images/"
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.summary()
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)
    history_dict = history.history

    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder + name + "_loss.png", dpi=150)  # uloží obrázek
    plt.show()

    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, acc, "bo-", label="Training accuracy")
    plt.plot(epochs, val_acc, "ro-", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder + name + "_accuracy.png", dpi=150)
    plt.show()

    return test_loss, test_acc


import time

model_data = []
default_activation = "relu"


def run(
    name,
    optimizer="adam",
    activation=default_activation,
    loss="binary_crossentropy",
    main_layers=[layers.GlobalAveragePooling1D()],
    extra_layers=None,
    max_tokens=10000,
    max_len=40,
    epochs=10,
    batch_size=32,
):
    if extra_layers is None:
        extra_layers = [layers.Dense(64, activation=activation)]
    start = time.time()
    print(f"-> Model {name}")
    model = keras.Sequential(
        [layers.Embedding(input_dim=max_tokens, output_dim=64, input_length=max_len)]
        + main_layers
        + extra_layers
        + [layers.Dropout(0.5), layers.Dense(1, activation="sigmoid")]
    )
    train_ds, val_ds, test_ds = create_datasets(
        model,
        max_tokens=max_tokens,
        max_len=max_len,
        batch_size=batch_size,
        loss=loss,
        optimizer=optimizer,
    )
    loss, acc = use_model(name, epochs, model, train_ds, val_ds, test_ds)
    end = time.time()
    took = end - start
    print(f"<- Model {name} took {took}ms")
    model_data.append({"name": name, "time": took, "loss": loss, "acc": acc})
    # 3 empty lines to separate models
    for _ in range(3):
        print()


run("default")

# Různé počty epoch
## reference: 10 epochs
## default

## zda se model učí už na začátku.
run("5-epochs", epochs=5)

## zda začne přeučení (val_loss začne růst).
run("20-epochs", epochs=20)

# Různé počty batch_size
## reference: 32 batches
## default

## menší batch → více update kroků, často lepší generalizace, ale pomalejší.
run("16-batches", batch_size=16)

## větší batch → rychlejší epochy, ale model se může učit hůř.
run("64-batches", batch_size=64)

# Různý počet vrstev
## Model A (velmi jednoduchý): Dense(1, sigmoid)
run("model_a", extra_layers=[])

## Model B (střední): Dense(64, relu) → Dense(1, sigmoid)
## default

## Model C (složitější): Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)
run(
    "model_c",
    extra_layers=[
        layers.Dense(64, activation=default_activation),
        layers.Dense(128, activation=default_activation),
    ],
)

# Různá konfigurace vrstev
## Konfigurace 1 (baseline): GlobalAveragePooling
## default

## Konfigurace 2 (1D CNN): Conv1D(64, kernel_size=3, activation=relu) → GlobalMaxPooling1D
run(
    "model_c",
    main_layers=[
        layers.Conv1D(64, kernel_size=3, activation=default_activation),
        layers.GlobalMaxPooling1D(),
    ],
)

# Různé aktivační funkce
## Konfigurace 1 (baseline): ReLU
## default

## Konfigurace 2: Tanh
run("tanh", activation="tanh")

# Různé ztrátové funkce
## Konfigurace 1 (baseline): binary_crossentropy
## default

## Konfigurace 2: hinge
run("hinge", loss="hinge")

# Různé optimalizační techniky
## Adam (baseline).
## default

## SGD s momentum
run("sgd", optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9))

## RMSprop
run("sgd", optimizer=keras.optimizers.RMSprop(learning_rate=0.001))


print(model_data)

results = pd.DataFrame(model_data)
print(results)
results.to_csv("model_results.csv", index=False)
