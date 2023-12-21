from pydoc import classname
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

# Load your dataset and perform necessary preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'D:/projectpsd/dataset_revisi',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'D:/projectpsd/dataset_revisi',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model Definition
model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Melihat hasil akurasi, presisi, dan recall
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Tampilkan hasil akhir
print("Training Accuracy:", train_acc[-1])
print("Validation Accuracy:", val_acc[-1])
print("Training Loss:", train_loss[-1])
print("Validation Loss:", val_loss[-1])

# Evaluate the model on the testing set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'D:/projectpsd/dataset_revisi',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model on the testing set
results = model.evaluate(test_generator)

# Extract metrics
test_loss = results[0]
test_accuracy = results[1]

# Make predictions on the test set
predictions = model.predict(test_generator)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
 


# Extract true labels
true_labels = test_generator.classes

# Save predictions and true labels for later use
np.save('predicted_labels.npy', predicted_labels)
np.save('true_labels.npy', true_labels)

# Save classification report
class_names = list(train_generator.class_indices.keys())
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
with open('classification_report.txt', 'w') as file:
    file.write(class_report)


# Tampilkan hasil
print("Testing Loss:", test_loss)
print("Testing Accuracy:", test_accuracy)

# Save the trained model
model.save('cnn1_model.h5')