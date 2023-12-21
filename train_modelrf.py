import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Assuming 'data' is a TensorFlow Dataset
data = tf.keras.utils.image_dataset_from_directory('D:/projectpsd/dataset_revisi', image_size=(256, 256), batch_size=32)

def main():
    # Convert the TensorFlow Dataset to NumPy arrays
    X, y = next(iter(data))

    # Flatten X
    X_flat = tf.reshape(X, (X.shape[0], -1))

    # Convert labels to NumPy arrays
    y = y.numpy()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_flat.numpy(), y, test_size=0.2, random_state=42)

    # Split data into k-folds using KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define the RandomForest model
    rf_model = RandomForestClassifier()

    # Define the list of parameters to be tested
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               scoring='accuracy', cv=kfold, n_jobs=-1)

    # Train the model using the data
    grid_search.fit(X_train, y_train)

    # Save the best model using joblib
    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, 'model.joblib')

    # Print the best parameters and best accuracy score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    # Predict using the best model
    y_pred = best_rf_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)

    print(classification_report(y_test, y_pred, zero_division=1))

if __name__ == "__main__":
    main()
