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


<p align="center">Made with 🔥 and PyTorch</p>
