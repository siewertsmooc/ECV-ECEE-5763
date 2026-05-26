import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# 1) Load and preprocess MNIST
# -----------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 -> 784
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("Train:", x_train.shape, y_train.shape)
print("Test: ", x_test.shape, y_test.shape)

# ---------------------------------------------------
# 2) Layer-wise unsupervised pretraining: Autoencoder 1
# ---------------------------------------------------
input_dim = 784
hidden1_dim = 256
hidden2_dim = 128

input_1 = keras.Input(shape=(input_dim,))
encoded_1 = layers.Dense(hidden1_dim, activation="sigmoid", name="encoder_1")(input_1)
decoded_1 = layers.Dense(input_dim, activation="sigmoid", name="decoder_1")(encoded_1)

autoencoder_1 = keras.Model(input_1, decoded_1, name="autoencoder_1")
autoencoder_1.compile(optimizer="adam", loss="binary_crossentropy")

autoencoder_1.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=2,
)

encoder_1 = keras.Model(input_1, encoded_1, name="pretrained_encoder_1")
x_train_h1 = encoder_1.predict(x_train, batch_size=256, verbose=0)
x_test_h1 = encoder_1.predict(x_test, batch_size=256, verbose=0)

# ---------------------------------------------------
# 3) Layer-wise unsupervised pretraining: Autoencoder 2
# ---------------------------------------------------
input_2 = keras.Input(shape=(hidden1_dim,))
encoded_2 = layers.Dense(hidden2_dim, activation="sigmoid", name="encoder_2")(input_2)
decoded_2 = layers.Dense(hidden1_dim, activation="sigmoid", name="decoder_2")(encoded_2)

autoencoder_2 = keras.Model(input_2, decoded_2, name="autoencoder_2")
autoencoder_2.compile(optimizer="adam", loss="binary_crossentropy")

autoencoder_2.fit(
    x_train_h1,
    x_train_h1,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=2,
)

encoder_2 = keras.Model(input_2, encoded_2, name="pretrained_encoder_2")

# ----------------------------------------
# 4) Build classifier with pretrained layers
# ----------------------------------------
classifier_input = keras.Input(shape=(input_dim,), name="mnist_input")

h1 = layers.Dense(hidden1_dim, activation="sigmoid", name="dbn_hidden_1")(classifier_input)
h2 = layers.Dense(hidden2_dim, activation="sigmoid", name="dbn_hidden_2")(h1)
output = layers.Dense(num_classes, activation="softmax", name="classifier_output")(h2)

dbn_classifier = keras.Model(classifier_input, output, name="dbn_style_classifier")

# Copy pretrained weights
dbn_classifier.get_layer("dbn_hidden_1").set_weights(
    autoencoder_1.get_layer("encoder_1").get_weights()
)
dbn_classifier.get_layer("dbn_hidden_2").set_weights(
    autoencoder_2.get_layer("encoder_2").get_weights()
)

# ----------------------------------------
# 5) Supervised fine-tuning
# ----------------------------------------
dbn_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = dbn_classifier.fit(
    x_train,
    y_train_cat,
    epochs=15,
    batch_size=256,
    validation_split=0.1,
    verbose=2,
)

# ----------------------------------------
# 6) Evaluate
# ----------------------------------------
test_loss, test_acc = dbn_classifier.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# ----------------------------------------
# 7) Example predictions
# ----------------------------------------
pred_probs = dbn_classifier.predict(x_test[:10], verbose=0)
pred_labels = pred_probs.argmax(axis=1)

print("True labels:     ", y_test[:10])
print("Predicted labels:", pred_labels)
