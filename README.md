

# 🔥 **README**

# SADN: Semantic-Aligned Decoupled Network for Few-Shot Object Detection

Official PyTorch implementation of our paper:
**“SADN: Semantic-Aligned Decoupled Network for Few-Shot Object Detection”**

---

## 🏗 Framework Overview 


<p align="center">
  <img src="assets/SADN.png" width="80%">
</p>
<p align="center">
  <img src="assets/HFRM.png" width="80%">
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



## 🧪 Training and Evaluation

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

| Dataset   | Model Weghts|
| :---: |:---: | 
| COCO-Base | [model](https://drive.google.com/file/d/1zxs66CXBOFDTFdMEa5v6ijIZSaORCSo7/view?usp=drive_link) | 
| VOC-Base1 | [model](https://drive.google.com/file/d/1zxs66CXBOFDTFdMEa5v6ijIZSaORCSo7/view?usp=drive_link) | 
| VOC-Base1 | [model](https://drive.google.com/file/d/1zxs66CXBOFDTFdMEa5v6ijIZSaORCSo7/view?usp=drive_link) | 
| VOC-Base1 | [model](https://drive.google.com/file/d/1zxs66CXBOFDTFdMEa5v6ijIZSaORCSo7/view?usp=drive_link)| 



---


---

## 📊 Experimental Results



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








