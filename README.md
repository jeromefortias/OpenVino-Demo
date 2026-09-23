# 🚀 Benchmark de Performance NER : PyTorch GPU (CUDA) vs OpenVINO CPU (FP16 / INT8 / INT4)

Ce dépôt contient une suite complète de benchmarks et d'outils d'analyse pour évaluer les performances d'inférence de modèles de reconnaissance d'entités nommées (**NER**), en comparant une exécution **PyTorch GPU Native (NVIDIA CUDA)** à différentes optimisations **Intel OpenVINO (CPU)**.

---

## 📺 Présentation Vidéo

Découvrez l'explication complète, la démonstration et l'analyse détaillée des résultats dans la vidéo YouTube dédiée :

[![Démo et Explication du Benchmark](https://img.youtube.com/vi/r_Il9e90A2s/maxresdefault.jpg)](https://www.youtube.com/watch?v=r_Il9e90A2s)

> 💡 **Regarder la vidéo sur YouTube :** [https://www.youtube.com/watch?v=r_Il9e90A2s](https://www.youtube.com/watch?v=r_Il9e90A2s)

---

## 📌 Présentation du Projet

L'objectif de ce benchmark est de mesurer la latence d'inférence et le coût temporel par jeton (*ms/token*) du modèle **`FacebookAI/xlm-roberta-large-finetuned-conll03-english`** à travers 4 moteurs d'exécution distincts :

1. **PyTorch GPU (CUDA)** : Modèle complet exécuté sur GPU NVIDIA.
2. **OpenVINO CPU (Base FP16)** : Modèle converti au format OpenVINO Intermediate Representation (IR).
3. **OpenVINO CPU (INT8 Quantized)** : Modèle quantifié en précision 8-bit avec **NNCF** (*Neural Network Compression Framework*).
4. **OpenVINO CPU (INT4 Compressed)** : Modèle compressé en précision 4-bit (*Weight Compression*).

Le projet inclut également des outils permettant d'extraire et de visualiser l'intégralité des poids numériques d'un modèle `.safetensors` ainsi qu'un collecteur automatique des métriques matérielles (`SystemInfo.json`).

---

## 📊 Graphiques de Performance Générés

Lors de l'exécution du benchmark (`testOpenVinovsGPU.py`), trois graphiques haute résolution sont automatiquement générés :

- **`average_performance_comparison_xlm_roberta_4way.png`** : Comparaison du coût moyen par jeton (ms/token) et de la latence moyenne par phrase (ms).
- **`latency_distribution_boxplot_xlm_roberta.png`** : Distribution de la latence par moteur sous forme de *boxplot*.
- **`token_scaling_scatter_xlm_roberta.png`** : Graphique de dispersion (*scatter plot*) montrant le passage à l'échelle de la latence selon la longueur de la phrase (nombre de jetons).

---

## 📂 Structure du Dépôt

| Fichier / Script | Description |
| :--- | :--- |
| **`testOpenVinovsGPU.py`** | Script principal du benchmark 4 voies (PyTorch CUDA vs OpenVINO FP16 / INT8 / INT4). Génère le fichier CSV et les graphiques. |
| **`SystemInfo.py`** | Exécute un diagnostic complet du système (CPU, RAM, GPU) et l'exporte au format `SystemInfo.json`. |
| **`ConvertSafeTensors2Json.py`** | Extrait l'ensemble des poids d'un fichier `.safetensors` et les sauvegarde sous forme d'un dictionnaire `model_raw_weights.json`. |
| **`ConvertSafeTensors2Text.py`** | Extrait les poids d'un fichier `.safetensors` sous la forme d'un fichier texte brut structuré (`model_raw_weights.txt`). |
| **`view_weight.py`** | Visualiseur interactif en ligne de commande pour parcourir les gros fichiers de poids texte (`model_raw_weights.txt`) sans surcharger la mémoire RAM. |
| **`lines.txt`** | Fichier texte contenant le jeu de phrases de test pour l'évaluation. |

---

## 🛠️ Installation & Prérequis

### 1. Cloner le projet
```bash
git clone https://github.com/votre-compte/votre-depot.git
cd votre-depot
```

### 2. Installer les dépendances
Il est recommandé d'utiliser un environnement virtuel Python (`venv` ou `conda`) :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers optimum[intel] openvino nncf pandas matplotlib seaborn psutil safetensors
```

---

## 🚀 Utilisation

### 1. Collecter les spécifications matérielles du système
Avant de lancer le benchmark, enregistrez les caractéristiques matérielles de votre machine :
```bash
python SystemInfo.py
```
*Génère un fichier `SystemInfo.json` contenant la version de l'OS, le modèle CPU, la RAM et les GPU détectés.*

### 2. Lancer le Benchmark 4-Voies
Assurez-vous d'avoir rempli le fichier `lines.txt` avec des phrases, puis lancez :
```bash
python testOpenVinovsGPU.py
```
*Le script va :*
1. Télécharger et stocker le modèle Hugging Face en local dans `./safetensors_xlm_roberta`.
2. Exporter et quantifier les variantes OpenVINO (FP16, INT8, INT4).
3. Exécuter l'inférence sur toutes les phrases pour chaque moteur.
4. Exporter les métriques brutes dans `benchmark_summary_xlm_roberta_4way.csv`.
5. Sauvegarder les 3 graphiques de résultats.

### 3. Exporter et inspecter les poids bruts du modèle (Optionnel)

- **Export au format JSON :**
  ```bash
  python ConvertSafeTensors2Json.py
  ```
- **Export au format Texte :**
  ```bash
  python ConvertSafeTensors2Text.py
  ```
- **Naviguer dans le fichier texte de poids via la console :**
  ```bash
  python view_weight.py
  ```
  *(Raccourcis : Flèches haut/bas, Page Suiv./Préc., Espace, Touche Q pour quitter).*

---

## 📄 Licence

Ce projet est sous licence MIT. N'hésitez pas à le fork, l'améliorer ou l'adapter à vos propres modèles d'IA !
