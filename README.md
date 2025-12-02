

# 🔥 **README**

# SADN: Semantic-Aligned Decoupled Network for Few-Shot Object Detection

Official PyTorch implementation of our paper:
**“SADN: Semantic-Aligned Decoupled Network for Few-Shot Object Detection”**

---

## 🏗 Framework Overview 
> 🔽 **请将你画的模型结构图放这里，如 `figures/framework.png`**

<p align="center">
  <img src="assert/sadn.png" width="80%">
</p>


---

## 📦 Installation

### 1. Clone this repo

```bash
git clone https://github.com/<your-github>/SADN.git
cd SADN
```

### 2. Create environment

```bash
conda create -n sadn python=3.8 -y
conda activate sadn
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Detectron2 (compatible version)

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## 📁 Project Structure

```
SADN/
│── configs/
│── datasets/
│── engine/
│── models/
│   │── CGDM/
│   │── TDC/
│   │── HFRM/
│── tools/
│   │── train_net.py
│   │── test_net.py
│── weights/
│── README.md
```

---

## ▶️ Training

### **1. Base training (Stage 1)**

```bash
python tools/train_net.py \
    --config-file configs/sadn_base.yaml \
    --num-gpus 4
```

### **2. Few-shot finetuning (Stage 2)**

```bash
python tools/train_net.py \
    --config-file configs/sadn_finetune.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS weights/sadn_base.pth
```

---

## 🧪 Evaluation

### **Evaluate on COCO**

```bash
python tools/test_net.py \
    --config-file configs/sadn_finetune.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS weights/sadn_ft.pth
```

### **Evaluate on VOC**

```bash
python tools/test_net.py \
    --config-file configs/sadn_voc.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS weights/sadn_voc.pth
```

---

## 📥 Pretrained Weights

| Model     | Dataset | Stage              | Download      |
| --------- | ------- | ------------------ | ------------- |
| SADN-Base | COCO    | Base training      | `<your-link>` |
| SADN      | COCO    | Few-shot finetuned | `<your-link>` |
| SADN      | VOC     | Few-shot finetuned | `<your-link>` |

请将 `<your-link>` 替换为你自己的权重地址（Google Drive / Baidu / GitHub Release）。

---


---

## 📊 Experimental Results (插图位)

> 🔽 **放你的 VOC/COCO 结果可视化或表格截图**

### COCO Few-Shot (nAP)

<p align="center">
  <img src="figures/coco_results.png" width="70%">
</p>

### VOC Few-Shot (AP50)

<p align="center">
  <img src="figures/voc_results.png" width="70%">
</p>

### Detection Visualization

<p align="center">
  <img src="figures/vis.png" width="80%">
</p>

---

## ✏️ Citation

如果你将其投稿 / 发表，放上 BibTeX：

```bibtex
@article{your_sadn_2025,
  title={SADN: Semantic-Aligned Decoupled Network for Few-Shot Object Detection},
  author={Li, Saibo and Wang, Yuxiang and ...},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This project is built on [Detectron2](https://github.com/facebookresearch/detectron2) and [DeFRCN](https://github.com/er-muyue/DeFRCN).


---

# 🎉 完成！

