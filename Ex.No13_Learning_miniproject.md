# Ex.No: 10 - User Identification Using Walking Activity with ResNet
### DATE: 01-11-2025                                                                           
### REGISTER NUMBER : 212222220048
### AIM: 
To write a program to train the classifier for user identification using walking activity data collected from smartphone accelerometers.

###  Algorithm:
```
1.Load and extract accelerometer data from all participants CSV files.
2.Segment the data using a sliding window to create fixed-length sequences.
3.Label each sequence based on the participant ID.
4.Split the dataset into training and test sets after one-hot encoding labels.
5.Build a 1D ResNet model with residual blocks for time-series classification.
6.Train the model on the training data using categorical cross-entropy loss.
7.Evaluate the model on test data and predict sample labels.
```

### Program:
```
# STEP 1: Unzip the Dataset
import zipfile, os
zip_path = "/content/user+identification+from+walking+activity.zip"
extract_path = "/content/data"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# STEP 2: Preprocess the Data
import glob
import numpy as np
import pandas as pd

all_files = glob.glob("/content/data/User Identification From Walking Activity/*.csv")
data, labels = [], []
window_size, stride = 100, 50

for i, file in enumerate(all_files):
    try:
        df = pd.read_csv(file, header=None)
        df.columns = ['time', 'x', 'y', 'z']
        df = df[['x', 'y', 'z']].astype('float32')
        if len(df) < window_size:
            continue
        for j in range(0, len(df) - window_size, stride):
            segment = df.iloc[j:j + window_size].values
            data.append(segment)
            labels.append(i)
    except Exception as e:
        print(f"Error with {file}: {e}")

X = np.array(data)
y = np.array(labels)

# STEP 3: Encode Labels and Split Dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# STEP 4: Build 1D ResNet Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense

def residual_block(x, filters, kernel_size):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 8, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 64, 5)
    x = residual_block(x, 64, 5)
    x = residual_block(x, 64, 5)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

model = build_resnet(input_shape=(X.shape[1], X.shape[2]), num_classes=y_cat.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# STEP 5: Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# STEP 6: Evaluate and Predict
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

sample = X_test[0].reshape(1, X.shape[1], X.shape[2])
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)
true_class = np.argmax(y_test[0])
print(f"Predicted Participant ID: {predicted_class}, True ID: {true_class}")
```

### Output:
![Screenshot 2025-05-19 085825](https://github.com/user-attachments/assets/d6f52612-63a1-4f60-b184-29d125ccb411)


### Result:
Thus the system was trained successfully and the prediction was carried out.
