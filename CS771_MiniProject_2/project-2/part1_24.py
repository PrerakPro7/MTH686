import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import torch

num_classes = 10

# The functions defined are indicated with their uses.

def load_dataset(filepath):
    """Loads a dataset from a given file path."""
    t = torch.load(filepath)
    data, targets = t['data'], t.get('targets') 
    return data, targets

def load_mobilenet(input_shape=(224, 224, 3)):
    """Load pre-trained MobileNet as a feature extractor."""
    return tf.keras.applications.MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)

def preprocess_data(data):
    """Preprocess raw image data to match MobileNet input requirements."""
    data = data.astype(np.float32)
    resized_data = np.array([tf.image.resize(img, (224, 224)).numpy() for img in data])
    return tf.keras.applications.mobilenet.preprocess_input(resized_data)

def extract_features(model, data):
    """Extract features using MobileNet."""
    preprocessed_data = preprocess_data(data)
    return model.predict(preprocessed_data, batch_size=32, verbose=1)

def reduce_features(data, n_components=256, fit_pca=None):
    """Perform PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components) if fit_pca is None else fit_pca
    return pca.fit_transform(data) if fit_pca is None else pca.transform(data), pca


# Description of the LwP CLassifier.

def initialize_prototypes(data, labels, num_classes):
    """Initialize prototypes for LwP classifier."""
    prototypes = []
    for cls in range(num_classes):
        cls_data = data[labels == cls]
        cls_prototype = cls_data.mean(axis=0)
        prototypes.append(cls_prototype)
    return np.array(prototypes)

def predict_labels(data, prototypes):
    """Predict labels for the given data using prototypes."""
    distances = cdist(data, prototypes)
    return np.argmin(distances, axis=1)

def update_prototypes(prototypes, data, pseudo_labels, num_classes, alpha=0.7):
    """Update prototypes using pseudo-labeled data."""
    for cls in range(num_classes):
        cls_data = data[pseudo_labels == cls]
        if len(cls_data) > 0:
            cls_mean = cls_data.mean(axis=0)
            prototypes[cls] = alpha * prototypes[cls] + (1 - alpha) * cls_mean
    return prototypes

# Major function that calls all the other functions defined above.

def train_and_evaluate(train_files, eval_files, num_classes=10, alpha=0.7, confidence_threshold=0.9):
    """
    Train models f1, ..., f10 and evaluate on held-out datasets.
    """
    accuracies = np.zeros((len(train_files), len(eval_files)))
    mobilenet = load_mobilenet()
    pca = None
    prototypes = None

    # Load and prepare D1
    data, targets = load_dataset(train_files[0])
    features = extract_features(mobilenet, data)
    reduced_features, pca = reduce_features(features, fit_pca=None)
    prototypes = initialize_prototypes(reduced_features, np.array(targets), num_classes)

    for i in range(1, len(train_files) + 1):
        print(f"Training model f{i}...")

        # Evaluate on held-out datasets
        for j in range(i):  
            eval_data, eval_targets = load_dataset(eval_files[j])
            eval_features = extract_features(mobilenet, eval_data)
            eval_reduced_features, _ = reduce_features(eval_features, fit_pca=pca)
            predictions = predict_labels(eval_reduced_features, prototypes)
            accuracies[i - 1, j] = accuracy_score(eval_targets, predictions)

        if i == len(train_files):
            break

        # Load next unlabeled dataset (D2, ..., D10)
        next_data, _ = load_dataset(train_files[i])
        next_features = extract_features(mobilenet, next_data)
        next_reduced_features, _ = reduce_features(next_features, fit_pca=pca)

        # Predict labels for next dataset
        pseudo_labels = predict_labels(next_reduced_features, prototypes)

        # Confidence filtering
        if confidence_threshold:
            distances = cdist(next_reduced_features, prototypes)
            confidence = 1 - (distances.min(axis=1) / distances.max(axis=1))
            mask = confidence >= confidence_threshold
            next_reduced_features = next_reduced_features[mask]
            pseudo_labels = pseudo_labels[mask]

        # Update prototypes using pseudo-labeled data
        prototypes = update_prototypes(prototypes, next_reduced_features, pseudo_labels, num_classes, alpha=alpha)

    return accuracies

# The code start from here. (Kindly provide the relevant paths below./ I have assigned path to my mounted drive.)

train_files = [f"/content/drive/MyDrive/CS771_mini/dataset/part_one_dataset/train_data/{i}_train_data.tar.pth" for i in range(1, 11)]  
eval_files = [f"/content/drive/MyDrive/CS771_mini/dataset/part_one_dataset/eval_data/{i}_eval_data.tar.pth" for i in range(1, 11)] 
accuracies = train_and_evaluate(train_files, eval_files)

# Printing final answer...

print("Accuracy matrix:")
print(accuracies)