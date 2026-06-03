# pip install scikit-learn matplotlib numpy
#
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

print("Avoid memory hogging")
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

RESULTS_DIR = "digits-results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN input
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model summary
model.summary()

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Test model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Save model
model.save("mnist_cnn.keras")
print("Saved model to mnist_cnn.keras")

# Reload model to verify save worked
loaded_model = tf.keras.models.load_model("mnist_cnn.keras")
loaded_loss, loaded_acc = loaded_model.evaluate(x_test, y_test, verbose=0)
print("Reloaded model accuracy:", loaded_acc)

# -------------------------
# Generate predictions
# -------------------------
y_prob = loaded_model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(ax=ax, colorbar=False)
ax.set_title("MNIST Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# -------------------------
# Precision-recall curves
# -------------------------
y_test_bin = label_binarize(y_test, classes=np.arange(10))
ap_scores = {}

fig, ax = plt.subplots(figsize=(10, 8))
for i in range(10):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])
    ap_scores[str(i)] = float(ap)
    ax.plot(recall, precision, label=f"Digit {i} (AP={ap:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves by Class")
ax.legend(loc="best", fontsize=8)
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "precision_recall_curves.png"), dpi=150)
plt.close(fig)

# -------------------------
# Classification report
# -------------------------
report_text = classification_report(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report_text)

with open(os.path.join(RESULTS_DIR, "classification_report.json"), "w") as f:
    json.dump(report_dict, f, indent=2)

# -------------------------
# Save training history
# -------------------------
history_dict = {
    "loss": [float(x) for x in history.history.get("loss", [])],
    "accuracy": [float(x) for x in history.history.get("accuracy", [])],
    "val_loss": [float(x) for x in history.history.get("val_loss", [])],
    "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
}

with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
    json.dump(history_dict, f, indent=2)

# -------------------------
# Save raw predictions
# -------------------------
np.savez(
    os.path.join(RESULTS_DIR, "predictions.npz"),
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
)

# -------------------------
# Save summary
# -------------------------
summary = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "reloaded_test_loss": float(loaded_loss),
    "reloaded_test_accuracy": float(loaded_acc),
    "macro_avg_precision": float(np.mean(list(ap_scores.values()))),
    "average_precision_per_class": ap_scores,
    "macro_f1": float(report_dict["macro avg"]["f1-score"]),
    "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
}

with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved digits-results:")
print(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
print(os.path.join(RESULTS_DIR, "precision_recall_curves.png"))
print(os.path.join(RESULTS_DIR, "classification_report.txt"))
print(os.path.join(RESULTS_DIR, "classification_report.json"))
print(os.path.join(RESULTS_DIR, "history.json"))
print(os.path.join(RESULTS_DIR, "predictions.npz"))
print(os.path.join(RESULTS_DIR, "summary.json"))
