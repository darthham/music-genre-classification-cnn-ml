# Music Genre Classification Using CNN and Classical Machine Learning Algorithms

## Project Team

| Name | SRN |
|---|---|
| Haneesh E | PES1UG24AM471 |
| Pranav Prabhakar | PES1UG24AM463 |

---

## 1. Project Overview

Music Genre Classification is a machine learning project that automatically identifies the genre of a music audio file using its acoustic characteristics.

This project implements a hybrid approach combining a **Convolutional Neural Network (CNN)** with classical machine learning algorithms to classify music into five genres:

- Blues
- Classical
- Hip-Hop
- Metal
- Pop

The system extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio signals. These features are used to train a CNN, which learns meaningful representations of music. The learned CNN features are then supplied to classical machine learning classifiers for comparison.

The objective is to evaluate whether CNN-extracted features can improve the performance of traditional classification algorithms.

## 2. Problem Statement

Identifying music genres manually can be time-consuming, especially when dealing with large music collections.

Traditional machine learning approaches depend heavily on manually extracted audio features. Deep learning models, on the other hand, can learn useful representations directly from input features.

This project explores the combination of CNN-based feature extraction and classical classification algorithms to improve music genre recognition.

## 3. Dataset

The project uses the GTZAN Music Genre dataset.

The implementation currently uses five selected genres:

| Genre | Number of audio files |
|---|---:|
| Blues | 200 |
| Classical | 200 |
| Hip-Hop | 200 |
| Metal | 200 |
| Pop | 200 |
| **Total** | **1,000** |

Audio files are processed as mono signals with a sampling rate of 22,050 Hz and a duration of 30 seconds.

The dataset is divided into:

- Training set: 80% (800 samples)
- Testing set: 20% (200 samples)

The split is stratified to maintain the genre distribution in both sets.

**Dataset directory structure:**

```text
archive/
└── genres/
    ├── blues/
    ├── classical/
    ├── hiphop/
    ├── metal/
    └── pop/
```

The preprocessing script supports `.wav` and `.au` audio files.

## 4. Methodology

The project follows these main stages:

### Stage 1: Audio Preprocessing

- Load audio files using Librosa.
- Convert audio into mono format.
- Resample audio to 22,050 Hz.
- Standardize audio duration to 30 seconds.

### Stage 2: Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio signal.

Configuration:

| Parameter | Value |
|---|---:|
| Sampling rate | 22,050 Hz |
| Audio duration | 30 seconds |
| Number of MFCCs | 20 |
| Hop length | 512 |
| Target frames | 1,290 |

Each audio sample is represented as a feature matrix of shape `(20, 1290)`.

### Stage 3: CNN Training

A one-dimensional dilated CNN is trained using the extracted MFCC features.

The CNN learns patterns in the temporal sequence of audio features.

The trained CNN is used for two purposes:

1. Direct music genre classification.
2. Extraction of intermediate CNN feature representations.

The standalone CNN achieved **93% test accuracy** in the current experiment.

### Stage 4: Feature Representation Comparison

Four feature representations are evaluated:

1. Raw MFCC features.
2. PCA-reduced features (50 components).
3. CNN Layer 1 features.
4. CNN Layer 2 features.

Global average pooling is applied to intermediate CNN feature maps to obtain fixed-length feature vectors.

### Stage 5: Classical Machine Learning Classification

The following classifiers are evaluated:

- Logistic Regression
- Support Vector Machine (SVM with RBF kernel)
- Random Forest
- Linear Discriminant Analysis (LDA)

Each classifier is evaluated using the different feature representations.

### Stage 6: Model Evaluation

Model performance is measured using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Accuracy comparison graphs and confusion matrices are generated to visualize the results.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | CNN implementation and training |
| Librosa | Audio loading and MFCC extraction |
| NumPy | Numerical operations |
| Scikit-learn | Classical ML algorithms and evaluation |
| Matplotlib | Graph generation |
| Seaborn | Confusion matrix visualization |
| Joblib | Saving and loading the scaler |

## 6. Project Structure

```text
music-genre-classification/
│
├── archive/
│   └── genres/
│       ├── blues/
│       ├── classical/
│       ├── hiphop/
│       ├── metal/
│       └── pop/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── prepare_data.py
│   ├── model.py
│   ├── train_cnn.py
│   ├── evaluate_classical.py
│   └── plot_results.py
│
├── artifacts/
│   ├── features/
│   │   └── mfcc_dataset.npz
│   │
│   └── models/
│       ├── dilated_cnn.pt
│       ├── mfcc_scaler.joblib
│       └── training_info.json
│
├── reports/
│   ├── generated/
│   │   ├── accuracy_comparison.png
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_best.png
│   │   └── classical_results.json
│   │
│   └── writeup_template.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Installation and Execution

### Prerequisites

- Python 3.10 or later
- pip
- Git
- VS Code (recommended)

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd music-genre-classification
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Add the dataset

Place the GTZAN audio dataset inside:

```text
archive/genres/
```

Ensure the five selected genre folders are present.

### Step 5: Preprocess the dataset

```bash
python -m src.prepare_data --data-dir archive/genres
```

This extracts MFCC features and creates the training and testing splits.

### Step 6: Train the CNN

```bash
python -m src.train_cnn
```

The trained model and feature scaler are saved in:

```text
artifacts/models/
```

### Step 7: Evaluate classical classifiers

```bash
python -m src.evaluate_classical
```

This evaluates the four classifiers across all four feature representations and generates evaluation results and confusion matrices.

### Step 8: Generate the accuracy comparison graph

```bash
python -m src.plot_results
```

The graph is saved as:

```text
reports/generated/accuracy_comparison.png
```

**Note:** Run the stages in the order shown above. The evaluation stage requires the preprocessed dataset, trained CNN, and saved scaler.

## 8. Experimental Results

The following results were obtained using the current 80:20 train-test split.

### Accuracy Comparison

| Feature Representation | Logistic Regression | SVM | Random Forest | LDA |
|---|---:|---:|---:|---:|
| Raw MFCC | 94% | 94% | 93% | 69.5% |
| PCA (50) | 87% | 88.5% | 93% | 79.5% |
| CNN Layer 1 | 94% | 93% | 95% | 93% |
| CNN Layer 2 | 93.5% | 93.5% | **96%** | 94% |

### Best Performing Combination

**CNN Layer 2 + Random Forest**

- Accuracy: 96%
- Correct predictions: 192
- Incorrect predictions: 8
- Test samples: 200

### Standalone CNN Performance

- Training epochs: 35
- Final test accuracy: 93%
- Device used: CPU

The hybrid CNN feature extraction and Random Forest classification approach achieved the highest accuracy among the evaluated combinations.

## 9. Conclusion

This project demonstrates the use of CNN-based feature extraction combined with classical machine learning algorithms for music genre classification.

The experimental results indicate that CNN Layer 2 features combined with Random Forest achieved the best performance, with 96% accuracy on the selected test set.

The results also demonstrate that learned CNN representations can provide useful information for traditional machine learning classifiers.

### Future Enhancements

- Extend classification to all ten GTZAN genres.
- Experiment with different CNN architectures.
- Evaluate additional audio features such as spectral contrast and chroma.
- Perform cross-validation and evaluate performance across multiple data splits.
- Develop a user interface that accepts an audio file and predicts its genre.
- Test the model on music from sources outside the GTZAN dataset.

---

## 10. References

1. GTZAN Music Genre Dataset.
2. Librosa: Audio and Music Signal Analysis in Python.
3. PyTorch Documentation.
4. Scikit-learn Documentation.
5. Research paper: *Combining CNN and Classical Algorithms for Music Genre Classification*.

