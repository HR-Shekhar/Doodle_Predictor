# 🖍️ Doodle Classifier

A fun and efficient machine learning app that classifies your doodle into one of the 19 pre-defined classes using a trained deep learning model.

## 🚀 Project Overview

This project is built using a subset of the [Google Quick, Draw! Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap). It allows users to draw or upload a doodle and get real-time predictions via a trained DNN model. The entire ML pipeline — from data sampling to model training and deployment — is included in this repository.

The goal was to build a scalable ML-powered doodle recognition tool with modular notebooks and a clean deployment-ready Streamlit interface.

---

## 📂 Project Structure

├── app.py # Streamlit app for hosting the doodle classifier
├── doodle_sampling.ipynb # Sampling + visualization from Google Quick Draw raw dataset
├── doodle_model.ipynb # Model training and preprocessing
├── requirements.txt # Required libraries (version-pinned for compatibility)
├── packages.txt # Do not change this - required by Hugging Face Spaces
├── data/ (numpy bitmap of each file)
└── README.md # This file


---

## 🧠 Model Overview

- **Architecture**: Deep Neural Network (DNN) using TensorFlow/Keras  
- **Input Shape**: 784 (28x28 flattened doodle image)  
- **Output Classes**: 19 custom-selected doodle classes  
- **Activation**: Softmax  
- **Loss**: Categorical Crossentropy  
- **Optimizer**: Adam  

---

## 🗃️ Dataset

- **Source**: [Google Quick, Draw! Numpy Bitmap Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap)
- **Classes Used**: 19 (custom-selected from the full set of 345)
- **Samples**: 30,000 samples per class (570,000 total)

> ⚠️ Download full class `.npy` files manually via the link above or search _"Google Quick, Draw! Dataset"_ and use the custom script in `doodle_sampling.ipynb` to extract samples.

---

## 📊 Notebooks Explained

### `doodle_sampling.ipynb`

- Loads raw `.npy` files of selected classes.
- Samples 30,000 examples from each.
- Provides visualization using matplotlib.
- Saves sampled datasets for model training.

### `doodle_model.ipynb`

- Loads and combines all sampled classes.
- Preprocesses the data (reshaping, normalizing, labeling).
- Trains a Deep Neural Network.
- Evaluates and saves the model.

---

## 🌐 Web App

### `app.py`

- Streamlit app to predict doodles from user input.
- Minimal UI for clean interaction.
- Hosted locally or on platforms like Hugging Face Spaces.

---

## 🛠️ Installation

### 1. Clone the repository


git clone https://github.com/<your-username>/doodle-classifier.git
cd doodle-classifier


### 2. Setup the environment

pip install -r requirements.txt

⚠️ Important:
Do not modify packages.txt or change the structure/content of requirements.txt unless you're sure the versions are compatible with Streamlit and TensorFlow. These are configured for compatibility and Hugging Face deployment.


### 3. ▶️ Running the App

streamlit run app.py

If you’re deploying on Hugging Face Spaces or another cloud platform, make sure to include the model file in the appropriate directory.


## 🤝 Contributing

Contributions, improvements, or suggestions are welcome!
Feel free to open an issue or submit a pull request.

🙋‍♂️ Author

### Made with ❤️ by Himanshu Shekhar
Connect with me on LinkedIn(https://www.linkedin.com/in/himanshu-shekhar-19040b317/) or check out my other projects!