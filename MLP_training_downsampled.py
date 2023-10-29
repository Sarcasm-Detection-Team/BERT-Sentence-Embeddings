#MLP training on downsampled data 
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns



data_df = pd.read_csv("raw_data_embeddings_final.csv") #read embeddings 

#extract features
embeddings = data_df.iloc[:, 0:768].values
sentences = data_df["title"].values
labels = data_df["target"].values

#scale the embeddings 
scaler = MinMaxScaler()
embeddings = scaler.fit_transform(embeddings)

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42) #split train/test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42) #split train/validation

#convert labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#built the model architecture , here MLP
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(768,)))
model.add(Dropout(0.3))  #dropout added after the first dense layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  #dropout added after the second dense layer
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#store training history for loss and accuracy 
history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

epochs = 10
for epoch in range(epochs):   #train the model for 10 epochs 
    print(f"Epoch {epoch + 1}/{epochs}")
    history_epoch = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val))
    
    #values are stored for each batch in the epoch
    history['loss'].extend(history_epoch.history['loss'])
    history['val_loss'].extend(history_epoch.history['val_loss'])
    history['accuracy'].extend(history_epoch.history['accuracy'])
    history['val_accuracy'].extend(history_epoch.history['val_accuracy'])

#evaluate the model 
loss, accuracy = model.evaluate(X_test, y_test)  #accuracy & loss
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

#make predictions 
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

#other metrics
precision = precision_score(y_test_classes, y_pred_classes)
recall = recall_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes)

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

cm = confusion_matrix(y_test_classes, y_pred_classes)

# plot confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("confusion_matrix_downsampled.png")
plt.show()

#save the weights of the model 
model.save_weights("model_weights_downsampled.h5")

#plot Training and Validation Loss/Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("figure_downsampled.png") #save the plots as a figure 

plt.show()