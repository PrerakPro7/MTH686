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

def compute_mmd(source, target, kernel='rbf', gamma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) between two distributions."""
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

    if len(source) == 0 or len(target == 0):
        return 0

    if kernel == 'rbf':
        K_ss = rbf_kernel(source, source, gamma=gamma)
        K_tt = rbf_kernel(target, target, gamma=gamma)
        K_st = rbf_kernel(source, target, gamma=gamma)
    elif kernel == 'linear':
        K_ss = linear_kernel(source, source)
        K_tt = linear_kernel(target, target)
        K_st = linear_kernel(source, target)
    else:
        raise ValueError("Unsupported kernel. Use 'rbf' or 'linear'.")

    n, m = len(source), len(target)
    mmd = K_ss.sum() / (n * n) + K_tt.sum() / (m * m) - 2 * K_st.sum() / (n * m)
    return mmd

def domain_adaptation(target_features, source_features, lambda_mmd=0.1):
    """
    Align target features to the source domain using MMD.
    Args:
        target_features: Features from the current dataset.
        source_features: Features from the previous dataset.
        lambda_mmd: Regularization parameter for MMD.
    Returns:
        Aligned target features.
    """
    # Calculate MMD between target and source features
    mmd_loss = compute_mmd(source_features, target_features)

    # Align target features using gradient descent
    aligned_features = target_features - lambda_mmd * mmd_loss
    return aligned_features

def update_prototypes_with_regularization(prototypes, features, labels, previous_features, alpha=0.7):
    """
    Update prototypes using confident features and regularization with previous features.
    Args:
        prototypes: Current prototypes.
        features: Confident features.
        labels: Pseudo-labels of confident features.
        previous_features: Features from previous datasets.
        alpha: Regularization strength.
    Returns:
        Updated prototypes.
    """
    for cls in range(len(prototypes)):
        cls_features = features[labels == cls]
        if len(cls_features) > 0:
            cls_mean = cls_features.mean(axis=0)
            regularized_mean = alpha * prototypes[cls] + (1 - alpha) * cls_mean
            prototypes[cls] = regularized_mean
    return prototypes

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

        reduced_features = next_reduced_features

    return accuracies, prototypes, reduced_features

def train_and_evaluate_part2(train_files, eval_files, prototypes, prev_features, num_classes=10, alpha=0.7, confidence_threshold=0.9):
    """
    Train models f11, ..., f20 and evaluate on held-out datasets.
    """
    accuracies = np.zeros((len(train_files), len(eval_files)))
    mobilenet = load_mobilenet()
    pca = None

    for i in range(1, len(train_files) + 1):
        print(f"Training model f{i+10}...")
        data, _ = load_dataset(train_files[i - 1])
        features = extract_features(mobilenet, data)
        reduced_features, pca = reduce_features(features, fit_pca=pca)

        aligned_features = domain_adaptation(reduced_features, prev_features)

        predictions = predict_labels(aligned_features, prototypes)

        if confidence_threshold:
            distances = cdist(aligned_features, prototypes)
            confidence = 1 - (distances.min(axis=1) / distances.max(axis=1))
            mask = confidence >= confidence_threshold
            reduced_features = aligned_features[mask]
            predictions = predictions[mask]

        prototypes = update_prototypes_with_regularization(prototypes, reduced_features, predictions, num_classes, alpha=alpha)

        for j in range(i + 10):
            eval_data, eval_targets = load_dataset(eval_files[j])
            eval_features = extract_features(mobilenet, eval_data)
            eval_reduced_features, _ = reduce_features(eval_features, fit_pca=pca)
            predictions = predict_labels(eval_reduced_features, prototypes)
            accuracies[i - 1, j] = accuracy_score(eval_targets, predictions)

        prev_features = reduced_features

    return accuracies


# The code start from here. (Kindly provide the relevant paths below./ I have assigned path to my mounted drive.)

train_files = [f"dataset/part_one_dataset/train_data/{i}_train_data.tar.pth" for i in range(1, 11)]  
eval_files = [f"dataset/part_one_dataset/eval_data/{i}_eval_data.tar.pth" for i in range(1, 11)] 
accuracies, prev_prototypes, prev_features = train_and_evaluate(train_files, eval_files)

# Printing final answer...

print("Accuracy matrix til f10:")
print(accuracies)

# From here, it will take prototypes as input f10, and generate for f11, f12, ..... f20.

train_files_2 = [f"dataset/part_two_dataset/train_data/{i}_train_data.tar.pth" for i in range(1, 11)]  
eval_files_2 = [f"dataset/part_two_dataset/eval_data/{i}_eval_data.tar.pth" for i in range(1, 11)] 
eval_files = eval_files + eval_files_2
accuracies = train_and_evaluate_part2(train_files_2, eval_files, prev_prototypes, prev_features)

# Printing final answer...

print("Accuracy matrix after f10:")
print(accuracies)