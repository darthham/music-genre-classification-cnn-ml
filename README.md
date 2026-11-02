# Music Genre Classification using Dilated CNN and Classical ML

This project implements the approach described in *Combining CNN and Classical Algorithms for Music Genre Classification*.

## Selected genres
- classical
- hiphop
- metal
- pop
- blues

The reference paper uses 500 GTZAN tracks (100 per selected genre) and an 80/20 train/test split. Your actual counts and scores will be reported from the dataset and experiments you run.

## Pipeline
1. Load 30-second WAV files from GTZAN.
2. Extract 20 MFCC coefficients and normalize each feature using training-set statistics.
3. Train a 1D dilated CNN over the MFCC time sequence.
4. Extract hidden activations from both convolutional blocks.
5. Train Logistic Regression, SVM, Random Forest, and Linear Discriminant Analysis (GDA-style shared-covariance classifier) on:
   - flattened MFCC features,
   - PCA-reduced MFCC features,
   - CNN block 1 activations,
   - CNN block 2 activations.
6. Evaluate accuracy, classification report, and confusion matrices.

## Dataset layout
Download/extract GTZAN and arrange it as:

```text
data/
  genres_original/
    blues/
    classical/
    ...
    hiphop/
    metal/
    pop/
```

Only the five selected genres are used. Each genre folder should contain `.wav` files.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
```powershell
python -m src.prepare_data --data-dir data/genres_original
python -m src.train_cnn
python -m src.evaluate_classical
```

Outputs are saved under `artifacts/` and `reports/`.

## Notes
- Do not claim the paper's reported accuracies as your own results.
- Run all experiments before completing the Results section of the write-up.
- For a fair comparison, use the same train/test indices for all feature sets and classifiers.
