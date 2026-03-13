# Image Caption Generator

An end-to-end deep learning project that automatically generates natural-language captions for images. It combines a **DenseNet201** CNN for image feature extraction with a **Bidirectional LSTM** for sequence generation, trained on the **Flickr8k** dataset and deployed as a web app using **Streamlit**.

---

## Demo

Upload any `.jpg` or `.png` image and the model will generate a descriptive caption for it.

---

## How It Works

1. **Feature Extraction** — DenseNet201 (pretrained on ImageNet, top layer removed) encodes each image into a 1920-dim feature vector.
2. **Caption Generation** — A Bidirectional LSTM takes the image features and a partial caption sequence, then predicts the next word.
3. **Inference** — Generation starts from `startseq` and stops when `endseq` is predicted or max length (34) is reached.

---

## Dataset

- **Flickr8k** — 8,091 images, each with 5 human-written captions
- **Vocabulary size:** 8,485 unique words
- **Max caption length:** 34 tokens

---

## Model Architecture

| Component | Details |
|---|---|
| Feature Extractor | DenseNet201 pretrained on ImageNet (output: 1920-dim vector) |
| Image input size | 224 × 224 × 3 |
| Sequence model | Bidirectional LSTM |
| Embedding | Learned word embeddings |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Callbacks | ModelCheckpoint, EarlyStopping, ReduceLROnPlateau |

---

## Project Structure

```
├── main.py                   # Streamlit web app
├── main.ipynb                # Full training notebook
├── model.keras               # Trained caption model
├── feature_extractor.keras   # DenseNet201 feature extractor
├── tokenizer.pkl             # Fitted Keras tokenizer
├── captions.txt              # Flickr8k captions
├── requirements.txt          # Python dependencies
├── .python-version           # Python 3.12 (for Streamlit Cloud)
└── input_images/             # Sample test images
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/shweta406/Image-Captioning-.git
cd Image-Captioning-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run main.py
```

---

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Matplotlib

---

## Deployment

Deployed on **Streamlit Community Cloud**.  
To deploy your own copy:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set branch to `master`, main file to `main.py`
4. Click Deploy
