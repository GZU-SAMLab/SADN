

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
git clone https://github.com/GZU-SAMLab/SADN.git
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

### 4. Install Detectron2 (0.3)

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```



## 🧪 Training and Evaluation

### **To reproduce the FSOD results on COCO**

```bash
bash run_coco_fsod.sh r101 8 sadn
```

### **To reproduce the FSOD results on VOC**

```bash
bash run_voc_fsod.sh r101 8 sadn
```

---

## 📥 Pretrained Weights

| Dataset   | Model Weghts|
| :---: |:---: | 
| COCO-Base | [model](https://pan.quark.cn/s/68e8530ee6c6) | 
| VOC-Base1 | [model](https://pan.quark.cn/s/2b0ed478f47d) | 
| VOC-Base2 | [model](https://pan.quark.cn/s/a918cc33f3bd) | 
| VOC-Base3 | [model](https://pan.quark.cn/s/cbe9b6e3fd8d)| 



---


---

## 📊 Experimental Results



### COCO Few-Shot (nAP)

<p align="center">
   <img width="704" height="305" alt="image" src="https://github.com/user-attachments/assets/e30cc9c6-1530-4d0e-8799-56fe0ca1b1f0" />


</p>

### VOC Few-Shot (mAP50)

<p align="center">
  <img width="701" height="305" alt="image" src="https://github.com/user-attachments/assets/2eb07f78-4924-47c6-8203-c151711c3af6" />


</p>

### Detection Visualization

<p align="center">
  <img src="assets/vis.png" width="80%">
</p>

---



---

## 🤝 Acknowledgements

This project is built on [Detectron2](https://github.com/facebookresearch/detectron2) and [DeFRCN](https://github.com/er-muyue/DeFRCN).


---

# 🎉 完成！








