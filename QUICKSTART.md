# ⚡ Quick Start Guide

## 🚀 Chạy Demo trong 5 phút

### Step 1: Mở Google Colab

1. Truy cập [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload file `demo_trimedagent_colab.ipynb`

### Step 2: Chọn GPU

1. Click **Runtime → Change runtime type**
2. Chọn **T4 GPU** (miễn phí)
3. Click **Save**

### Step 3: Chạy Notebook

1. Click **Runtime → Run all**
2. Đợi cài đặt packages (~2-3 phút)
3. Đợi load models (~3-5 phút)
4. Sử dụng Gradio UI!

---

## 📦 Dependencies

```bash
# Core
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
bitsandbytes>=0.41.0

# Vision
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0

# Tools
segment-anything
groundingdino-py

# UI
gradio>=4.0.0

# RAG (optional)
groq
sentence-transformers
```

---

## 🔧 Troubleshooting

### Lỗi "CUDA out of memory"

```python
# Dùng 4-bit quantization
llava = LLaVATool(quantize_4bit=True)
```

### Lỗi Gradio asyncio

```python
# Thêm ở đầu notebook
import nest_asyncio
nest_asyncio.apply()
```

### Không có GPU

```python
# Chỉ dùng BiomedCLIP (nhẹ nhất)
clip = BiomedCLIPTool(device="cpu")
```

---

## 📚 Next Steps

1. Đọc [README.md](README.md) để hiểu kiến trúc
2. Xem code trong `models/` folder
3. Thử với ảnh y tế của bạn!
