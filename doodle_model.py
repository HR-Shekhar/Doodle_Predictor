#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers


# In[58]:


data_dir = ('./')
class_names = ["circle", "crown", "skull", "smiley_face", "square", "star"]


# In[59]:


samples_per_class = 5000

X = []
y = []


# In[60]:


for label, class_name in enumerate(class_names):
    # Create the path to the .npy file (e.g., './data/apple.npy')
    file_path = os.path.join(data_dir, f"{class_name}.npy")

    # Load the .npy file → shape: (2000, 28, 28)
    data = np.load(file_path)

    X.append(data)

    # Add labels to y → [label, label, label, ..., label] (length = number of samples)
    y.append(np.full((data.shape[0]), label))


# In[61]:


# Stack all image arrays vertically → final shape: (total_samples, 28, 28)
X = np.vstack(X)

# Stack all label arrays horizontally → final shape: (total_samples,)
y = np.hstack(y)


# ### Max Normalization

# In[62]:


# Normalize pixel values to range 0–1 (from 0–255)
X = X.astype('float32') / 255.0
X.shape


# In[63]:


# Flatten each image from 28x28 → 784 for dense layers
X = X.reshape(X.shape[0], -1)  # shape becomes (total_samples, 784)


# In[64]:


X_train, X_, y_train, y_ = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
# `stratify=y` ensures equal class distribution in both training and validation sets.

X_cv, X_test, y_cv, y_test = train_test_split(X_,y_, test_size=0.5, random_state=42, stratify=y_)


# In[ ]:


model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(len(class_names), activation='linear')
])


# In[66]:


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)


# In[67]:


model.fit(X_train,y_train, epochs=90)


# In[68]:


y_cv_pred_probs = model.predict(X_cv)   #predict probability for each label
y_cv_pred = np.argmax(y_cv_pred_probs, axis=1)  # selects the label with maximum probability


# In[69]:


print(accuracy_score(y_cv,y_cv_pred))


# In[70]:


y_test_pred_probs = model.predict(X_test) 
y_test_pred = np.argmax(y_test_pred_probs, axis=1)
print(accuracy_score(y_test,y_test_pred))


# In[72]:


model.save("doodle_model.keras")

