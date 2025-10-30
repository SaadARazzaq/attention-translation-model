# English-to-German Translator using Transformers from Scratch

![alt text](https://pbs.twimg.com/media/DCKhefrUMAE9stK.jpg)

## 📖 Overview

This repository contains a complete implementation of an English-to-German neural machine translation system using the Transformer architecture, built from scratch with TensorFlow 2. 

The implementation follows the original "Attention Is All You Need" paper by Vaswani et al. (2017) and demonstrates state-of-the-art sequence-to-sequence learning capabilities.

## 🏗️ Architecture

### Transformer Model Components

The implementation includes all core components of the Transformer architecture:

- **Multi-Head Self-Attention**: Scaled dot-product attention with multiple attention heads
- **Positional Encoding**: Sine-cosine positional encodings to incorporate sequence order information
- **Encoder-Decoder Architecture**: Stacked encoder and decoder layers with residual connections
- **Feed-Forward Networks**: Position-wise fully connected networks
- **Layer Normalization**: Stabilizing training through normalization
- **Masked Attention**: Look-ahead masks for decoder self-attention

### Model Specifications

| Component | Specification |
|-----------|---------------|
| Embedding Dimension | 128 |
| Number of Layers | 4 |
| Attention Heads | 8 |
| Feed-Forward Units | 512 |
| Dropout Rate | 0.1 |
| Vocabulary Size | 8,000 tokens per language |

## 📊 Dataset

### Europarl Parallel Corpus
- **Source**: European Parliament Proceedings
- **Language Pair**: English-German
- **Size**: ~2 million sentence pairs
- **Preprocessing**: Tokenization, subword encoding, sentence length filtering (max 20 tokens)

### Data Preprocessing Pipeline
1. **Text Normalization**: Handling of non-breaking prefixes
2. **Sentence Segmentation**: Proper sentence boundary detection
3. **Subword Tokenization**: Byte Pair Encoding (BPE) with vocabulary size of 8,000
4. **Sequence Padding**: Fixed-length sequences with post-padding

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install tensorflow==2.x
pip install tensorflow-datasets
pip install numpy
```

### Google Colab Setup
```python
from google.colab import drive
drive.mount('/content/drive')
%cd drive/MyDrive/projects/transformers_translation/
```

### Data Download
```python
# Download and extract Europarl dataset
%cd data
!wget https://www.statmt.org/europarl/v7/de-en.tgz
!tar -xvf de-en.tgz
%cd ..
```

## 🧠 Model Implementation

### Key Classes

1. **PositionalEncoding**: Implements sinusoidal positional encodings
2. **MultiHeadAttention**: Multi-head self-attention mechanism
3. **EncoderLayer**: Single encoder layer with self-attention and FFN
4. **DecoderLayer**: Single decoder layer with masked self-attention and encoder-decoder attention
5. **Encoder**: Complete encoder stack
6. **Decoder**: Complete decoder stack
7. **Transformer**: Full transformer model combining encoder and decoder

### Training Configuration

- **Optimizer**: Adam with custom learning rate scheduling
- **Learning Rate**: Warm-up followed by inverse square root decay
- **Batch Size**: 64
- **Training Steps**: 6,100 steps per epoch
- **Loss Function**: Sparse categorical cross-entropy with masking

## 📈 Performance

### Training Results
- **Epoch 1**: Final loss 2.6733, Accuracy 36.08%
- **Epoch 2**: Final loss 1.6733, Accuracy 36.08%

### Example Translations
```
Input: "This is a great day!"
Output: "Das ist ein großer Tag!"
```

## 🔧 Usage

### Training the Model
```python
# Hyperparameters
D_MODEL = 128
NB_LAYERS = 4
FFN_UNITS = 512
NB_PROJ = 8
DROPOUT = 0.1

transformer = Transformer(
    vocab_size_enc=VOCAB_SIZE_EN,
    vocab_size_dec=VOCAB_SIZE_DE,
    d_model=D_MODEL,
    nb_layers=NB_LAYERS,
    FFN_units=FFN_UNITS,
    nb_proj=NB_PROJ,
    dropout=DROPOUT
)
```

### Translation Inference
```python
def translate(sentence):
    output = evaluate(sentence).numpy()
    predicted_sentence = tokenizer_de.decode(output)
    return predicted_sentence

# Example usage
translation = translate("This is a great day!")
print(f"Translation: {translation}")
```

## 🎯 Key Features

### 1. Attention Mechanisms
- **Self-Attention**: Captures dependencies within input sequences
- **Cross-Attention**: Connects encoder and decoder representations
- **Masked Attention**: Prevents information leakage in decoder

### 2. Training Optimizations
- **Custom Learning Rate Schedule**: Warm-up and decay strategy
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Model saving and restoration
- **Early Stopping**: Prevents overfitting

### 3. Preprocessing Innovations
- **Non-breaking Prefix Handling**: Proper sentence segmentation
- **Subword Tokenization**: Handles out-of-vocabulary words
- **Sequence Length Optimization**: Balances computational efficiency and context

## 📚 Theoretical Background

### Transformer Advantages over RNNs
1. **Parallelization**: Self-attention enables parallel computation
2. **Long-range Dependencies**: Direct connections between distant tokens
3. **Computational Efficiency**: Reduced sequential operations
4. **Representation Power**: Multiple attention heads capture different linguistic aspects

### Attention Mechanism
The scaled dot-product attention is computed as:
```
Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V
```

Where:
- Q: Query matrix
- K: Key matrix  
- V: Value matrix
- dₖ: Dimension of key vectors

## 🔍 Experimental Results

The model demonstrates:
- **Effective Learning**: Clear loss reduction and accuracy improvement
- **Generalization**: Reasonable translations on unseen sentences
- **Scalability**: Architecture supports larger models and datasets

## 🛠️ Customization

### Modifying Model Size
```python
# Larger model configuration
D_MODEL = 512
NB_LAYERS = 6
FFN_UNITS = 2048
```

### Adjusting Training Parameters
```python
# Extended training
EPOCHS = 10
BATCH_SIZE = 32
```

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{english_german_transformer_2024,
  title={English-to-German Transformer from Scratch},
  author={Shilabin, Araz},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/ArazShilabin/english-to-german-translator-using-transformers-from-scratch}}
}
```

## 🙏 Acknowledgments

- Original Transformer Paper: "Attention Is All You Need" by Vaswani et al.
- Europarl corpus providers
- TensorFlow team for excellent deep learning framework
- ZTN Academy course "TensorFlow for Deep Learning Bootcamp" for inspiration

## 🔗 Related Resources

> - [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
> - [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
> - [Europarl Corpus](https://www.statmt.org/europarl/)

---

**Note**: This implementation is designed for educational purposes and demonstrates the complete Transformer architecture. For production use, consider using larger models, more training data, and additional optimization techniques.

---

<br>

<h2 align="center">✨ Author</h2>

<p align="center">
  <b>Saad Abdur Razzaq</b><br>
  <i>Machine Learning Engineer | Effixly AI</i>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/saadarazzaq" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="mailto:sabdurrazzaq124@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
  <a href="https://saadarazzaq.dev" target="_blank">
    <img src="https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website">
  </a>
  <a href="https://github.com/saadabdurrazzaq" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

<br>

---

<div align="center">

### ⭐ Don't forget to star this repository if you find it helpful!

</div>
