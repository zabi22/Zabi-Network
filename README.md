<p align="center">
  <img src="logo.svg" alt="Zabi Network Logo" width="200"/>
</p>

<h1 align="center">Zabi Network</h1>
<p align="center">AI image recognition platform built with PyTorch</p>

<p align="center">
  <a href="https://zabi-network.onrender.com">
    <img src="https://img.shields.io/badge/Live%20Demo-Render-7c3aed?style=for-the-badge&logo=render" alt="Live Demo"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-06b6d4?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  Train custom AI models through a sleek web dashboard. Upload your images, watch training in real-time, and get instant predictions.
</p>

---

## ✨ Features

- 🔥 **Hybrid Architecture** - CNN + RNN + Transformer combined for powerful image recognition
- 📊 **Live Training Dashboard** - Watch metrics update in real-time with beautiful charts
- 📤 **Drag & Drop Upload** - Upload your own datasets by class (dogs, cats, whatever)
- 🎯 **Instant Predictions** - Drop any image after training and get classifications
- ⚙️ **Fully Configurable** - Tweak learning rate, batch size, optimizer, epochs, etc.
- 🧪 **Synthetic Data Mode** - Test with auto-generated data before using real images
- 💾 **Auto Checkpointing** - Models save automatically, resume training anytime
- 🌐 **Deploy Anywhere** - Works on Render, Railway, or your local machine

---

## 🚀 Live Demo

Check it out: **[https://zabi-network.onrender.com](https://zabi-network.onrender.com)**

---

## ⚡ Quick Start

### Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the web app

```bash
python app.py
```

Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 📸 Train a Model

1. **Upload images** by class (e.g., "dogs", "cats", "cars") using the dataset section
2. **Adjust training settings** if needed (learning rate, epochs, batch size)
3. Click **"Start Training"**
4. **Watch live metrics** update in real-time on the dashboard
5. **After training**, upload any image to the Predict section to classify it

That's it! Your model is ready.

---

## 📂 Main Files

These are the 4 files you'll use most:

| File | What it does |
|------|-------------|
| **`app.py`** | Web server - runs the dashboard |
| **`model.py`** | The neural network (CNN + RNN + Transformer) |
| **`data.py`** | Loads your images for training |
| **`config.yaml`** | Change settings like learning rate, batch size, etc |

The other files in `core/` and `src/` handle training loops, metrics, custom layers, and utilities. You usually don't need to touch them.

---

## 🏗️ Architecture

The model combines three types of neural networks:

1. **CNN** - Extracts visual features from images
2. **Bidirectional LSTM** - Processes features as sequences  
3. **Transformer** - Applies self-attention with gating mechanisms

All core components (layer normalization, dropout, multi-head attention) are implemented **from scratch** without relying on high-level PyTorch modules. This gives you full control and deeper understanding of how everything works under the hood.

---

## 🎓 Training Options

You can train on:

- **Your own images** - Upload through the web interface or point to a folder
- **Synthetic data** - Random generated data for testing (default)

### Dataset Format

Organize your images like this:

```
your_dataset/
├── dogs/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── cats/
│   ├── img1.jpg
│   └── ...
└── birds/
    ├── img1.jpg
    └── ...
```

Each folder name becomes a class. The system automatically detects classes from subdirectories.

---

## 💻 CLI Usage

Prefer the command line? Got you:

```bash
# Train from command line
python main.py --mode train --config config.yaml

# Evaluate a trained model
python main.py --mode eval --resume ./checkpoints/best_model.pt

# Profile model performance
python main.py --mode profile

# Neural architecture search
python main.py --mode nas
```

---

## 🔧 Configuration

Edit `config.yaml` to change:

```yaml
model:
  num_classes: 10
  input_height: 32
  input_width: 32
  cnn_channels: [32, 64, 128, 256]
  rnn_type: "lstm"
  attn_num_heads: 8
  
train:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adamw"
  
data:
  augmentation: true
```

---

## 🌍 Deploy to Render

1. Fork this repo
2. Go to [Render.com](https://render.com)
3. Create a new Web Service
4. Connect your repo
5. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
6. Deploy!

You'll get a public URL anyone can access.

---

## 📋 Requirements

- Python 3.9+
- PyTorch
- Flask
- Pillow
- NumPy
- Matplotlib
- PyYAML

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
Zabi-Network/
├── app.py              # Flask web server (main entry point)
├── main.py             # CLI interface
├── config.yaml         # Default configuration
├── requirements.txt    # Dependencies
├── logo.svg            # Neural network logo
├── core/               # Core components
│   ├── config.py       # Configuration management
│   ├── layers.py       # Custom neural network layers
│   ├── losses.py       # Loss functions
│   ├── metrics.py      # Evaluation metrics
│   ├── trainer.py      # Training loop
│   └── utils.py        # Utilities (checkpointing, logging)
├── src/                # Source modules
│   ├── data.py         # Data loading and augmentation
│   └── model.py        # Main model architecture
└── templates/
    └── index.html      # Web dashboard UI
```

---

## 🤝 Contributing

Found a bug? Want to add features? Open an issue or submit a PR. All contributions welcome!

---

## 📄 License

MIT License - do whatever you want with this.

---

<p align="center">Made with 🔥 and PyTorch</p>
