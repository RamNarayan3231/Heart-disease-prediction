# train_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_and_save_models():
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv('heart.csv')
    print(f"Dataset loaded with {len(df)} records")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Scaler saved as 'scaler.pkl'")
    
    # Train Random Forest (Best ML model)
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate RF
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"✓ Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Save ML model
    joblib.dump(rf_model, 'best_ml_model.pkl')
    print("✓ ML model saved as 'best_ml_model.pkl'")
    
    # Create and train Deep Learning model
    print("Training Deep Learning model...")
    
    def create_dl_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    dl_model = create_dl_model()
    
    # Train the model
    history = dl_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate DL model
    y_pred_proba = dl_model.predict(X_test_scaled, verbose=0)
    y_pred_dl = (y_pred_proba > 0.5).astype(int).flatten()
    dl_accuracy = accuracy_score(y_test, y_pred_dl)
    print(f"✓ Deep Learning Accuracy: {dl_accuracy:.4f}")
    
    # Save DL model
    dl_model.save('best_dl_model.h5')
    print("✓ DL model saved as 'best_dl_model.h5'")
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Deep Learning Accuracy: {dl_accuracy:.4f}")
    
    best_model = "Random Forest" if rf_accuracy >= dl_accuracy else "Deep Learning"
    print(f"\n🎯 Best Model: {best_model}")
    
    # Create some visualizations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest - Confusion Matrix')
    
    plt.subplot(1, 3, 2)
    cm_dl = confusion_matrix(y_test, y_pred_dl)
    sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Blues')
    plt.title('Deep Learning - Confusion Matrix')
    
    plt.subplot(1, 3, 3)
    models = ['Random Forest', 'Deep Learning']
    accuracies = [rf_accuracy, dl_accuracy]
    plt.bar(models, accuracies, color=['blue', 'orange'])
    plt.title('Model Accuracies')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison plot saved as 'model_comparison.png'")
    
    print("\n✅ All models trained and saved successfully!")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    train_and_save_models()