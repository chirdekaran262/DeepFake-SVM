import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_model(X, y, model_dir="models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        raise ValueError("At least 2 classes are required.")

    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Insufficient class samples. Training on all data.")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kernels = ['linear', 'rbf', 'poly']
    best_model, best_accuracy, best_kernel = None, 0, None

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

        for kernel in kernels:
            model = SVC(kernel=kernel, probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            acc = accuracy_score(y_test, model.predict(X_test_scaled))
            print(f"Kernel={kernel}, Accuracy={acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_kernel = kernel

        print("\nBest Kernel:", best_kernel)
        print("Classification Report:\n", classification_report(y_test, best_model.predict(X_test_scaled)))
        print("Confusion Matrix:\n", confusion_matrix(y_test, best_model.predict(X_test_scaled)))
    else:
        best_model = SVC(kernel='linear', probability=True, random_state=42)
        best_model.fit(X_train_scaled, y_train)

    model_path = os.path.join(model_dir, "enhanced_svm_model.pkl")
    scaler_path = os.path.join(model_dir, "enhanced_scaler.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return best_model, scaler
