

# Readme

This project implements a machine learning pipeline for binary classification.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Understanding Important Functions](#understanding-important-functions)
    - [train_emoticon_model](#train_emoticon_model)
    - [train_text_seq_model](#train_text_seq_model)
    - [train_feat_model](#train_feat_model)
    - [train_combined_model](#train_combined_model)
    - [make_predictions](#make_predictions)
- [Training the Models](#training-the-models)
- [Running the Script](#running-the-script)
- [Output Files](#output-files)
- [Files and Directory Structure](#files-and-directory-structure)
- [Contact](#contact)


## Prerequisites

Before running the project, ensure the following:
1. **Python** (version 3.6 or higher) is installed.
2. Install the necessary Python libraries by running:
   ```bash
   pip install numpy pandas lightgbm tensorflow scikit-learn
   ```

3. Datasets should be available in the following format:
   - **Emoticon Data**: `.csv` files containing emoticon sequences.
   - **Text Data**: `.csv` files containing text sequences.
   - **Feature Data**: `.npz` files containing feature-based data.

## Project Setup

1. **Download the Project Files**:
   - Make sure you have the Python script (`24.py`) and the datasets.
   
2. **Organize the Project Directory**:
   Your directory should be organized as follows:
   ```
   project_directory/
   ├── 24.py                # The Python script
   ├── datasets/
       ├── train/           # Training data
       ├── valid/           # Validation data
       ├── test/            # Testing data
   ```

3. **Dataset Formats**:
   - **Training, Validation, and Test Data** should be split and stored in separate directories (`train`, `valid`, `test`).
   - The **datasets** directory should contain the following:
     - `train_emoticon.csv`, `val_emoticon.csv`, `test_emoticon.csv`
     - `train_text_seq.csv`, `val_text_seq.csv`, `test_text_seq.csv`
     - `train_feature.npz`, `valid_feature.npz`, `test_feature.npz`

## Understanding Important Functions

The script contains several functions that handle training, predictions, and data processing. Below are the key functions:

### 1. `train_emoticon_model(df_train_emoticon, df_val_emoticon, df_test_emoticon)`

**Purpose**: Trains an LSTM model on emoticon data.
- **Input**: CSV files containing emoticon sequences.
- **Output**: Trained model and predictions for emoticon data.

### 2. `train_text_seq_model(df_train_seq, df_val_seq, df_test_seq)`

**Purpose**: Trains an LSTM model on text sequences.
- **Input**: CSV files containing text sequences.
- **Output**: Trained model and predictions for text data.

### 3. `train_feat_model(train_feat, val_feat, test_feat)`

**Purpose**: Trains a LightGBM model on extracted features.
- **Input**: `.npz` files containing feature data.
- **Output**: Trained model and predictions for feature-based data.

### 4. `train_combined_model(X_train_emoticon, X_train_feat, X_train_seq, y_train_emoticon)`

**Purpose**: Combines predictions from the three models (Emoticon, Text, and Feature) to train the final combined model.
- **Input**: Predictions from the individual models.
- **Output**: Combined model and predictions on combined test data.

### 5. `make_predictions()`

**Purpose**: Uses trained models to generate predictions on the test datasets.
- **Output**: Saves predictions to text files (`.txt` format).

## Training the Models

The training process involves running all three individual models (emoticon, text sequence, and feature-based models), and then using their predictions for the combined model.

1. **Train the Emoticon Model**:
   - This model processes emoticon characters and removes irrelevant characters.
   - It uses an LSTM network to classify emoticon data.
   
2. **Train the Text Sequence Model**:
   - This model uses LSTM layers to classify text sequences.
   
3. **Train the Feature-based Model**:
   - This model uses a LightGBM framework to classify data based on feature vectors.

After training all three models, their predictions are passed into the **Combined Model** for final classification.

## Running the Script

Once the datasets are prepared and the project setup is complete, you can run the script:

1. **Step 1**: Open a terminal or command prompt in your project directory.

2. **Step 2**: Run the script using the following command:
   ```bash
   python 24.py
   ```

3. **Step 3**: The script will:
   - Preprocess the datasets (removing characters, encoding sequences, scaling features).
   - Train the individual models.
   - Combine their predictions and train the final combined model.
   - Save the predictions in `.txt` files.

## Output Files

Once the script has been executed, the following output files will be generated:

1. **`pred_emoticon.txt`**: Contains the predictions from the emoticon model on the test data.
2. **`pred_textseq.txt`**: Contains the predictions from the text sequence model on the test data.
3. **`pred_deepfeat.txt`**: Contains the predictions from the feature-based model on the test data.
4. **`pred_combined.txt`**: Contains the predictions from the combined model (final predictions).

Each file contains binary classification results (0 or 1) based on the model's predictions.

## Files and Directory Structure

The structure of the project should look like this:

```
project_directory/
├── 24.py
├── datasets/
│   ├── train/
│   │   ├── train_emoticon.csv
│   │   ├── train_text_seq.csv
│   │   ├── train_feature.npz
│   ├── valid/
│   │   ├── val_emoticon.csv
│   │   ├── val_text_seq.csv
│   │   ├── valid_feature.npz
│   ├── test/
│       ├── test_emoticon.csv
│       ├── test_text_seq.csv
│       ├── test_feature.npz
├── pred_emoticon.txt
├── pred_textseq.txt
├── pred_deepfeat.txt
├── pred_combined.txt
```


