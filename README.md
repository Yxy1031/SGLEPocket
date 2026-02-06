# SGLEPocket

Predicting protein-ligand binding pockets is crucial for understanding various biological processes, drug discovery, and design. Existing methods predominantly convert proteins into 3D voxels and process them using extensive convolutions, yet they often struggle to effectively capture long-range semantic information within proteins. Furthermore, the lack of global modeling and adaptive filtering of cross-layer features limits the precise characterization of detailed pocket features.

To tackle these issues, we propose a novel U-shaped network architecture that integrates spatial gating mechanisms and local feature enhancement for accurate protein-ligand binding pocket prediction. Specifically, we improve the traditional U-shaped network encoder by integrating the Mamba module and a Local Feature Enhancement (LFE) module to achieve efficient global modeling and adaptive enhancement of local features. Additionally, we introduce a novel Spatial Enhanced Mamba Gate (SEMG) module at skip connections to filter redundant information and enhance multiscale feature fusion. Experiments across extensive protein-ligand datasets demonstrate that our approach outperforms existing methods in both performance and interpretability.

---

## Datasets

### Train Data
* **scPDB**: [Download Link](http://bioinfo-pharma.u-strasbg.fr/scPDB/)

### Test Data
Test datasets can be downloaded from the following links:

* **COACH420**: [Download Link](https://github.com/rdk/p2rank-datasets/tree/master/coach420)
* **HOLO4k**: [Download Link](https://github.com/rdk/p2rank-datasets/tree/master/holo4k)
* **SC6K**: [Download Link](https://github.com/devalab/DeepPocket)
* **PDBbind**: [Download Link](http://www.pdbbind.org.cn/download.php)

---

## Environment Setup

### Dependencies
The following packages are required to run this project:

**Bioinformatics & Molecular Processing:**
* `biopython`
* `prody`
* `openbabel`
* `molgrid`
* `fpocket`

**Deep Learning:**
* `monai`
* `mamba-ssm`

**Scientific Computing:**
* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `scikit-image`

---

## Data Preprocessing


You can create your own preprocessed data by referring to the preprocessing steps used in **[DeepPocket](https://github.com/devalab/DeepPocket)**.

We provide the specific processing workflow in `predict_demo`:
> **Protein Cleaning** → **FPocket Candidate Pocket Identification** → **Candidate Center Extraction** → **Data Format Conversion** → **Pocket Ranking** → **Pocket Segmentation Prediction**


You can also directly use our pre-processed datasets:
* **Download Link**: [Baidu Netdisk](https://pan.baidu.com/s/1__432kgs0ZN2XfLKvbc50Q)
* **Password**: `1v7s`

---

## Voxelization Process

The specific voxelization modeling process can be found in the project data processing section: `Dataset/Protein_Dataset.py`

### Implementation Details
We use `libmolgrid` for processing to generate the 65 × 65 × 65 voxel grid, with the following code configuration:

```python
self.gmaker_mask = molgrid.GridMaker(dimension=32, binary=True, gaussian_radius_multiple=-1, resolution=0.5)
self.gmaker_img = molgrid.GridMaker(dimension=32, radius_scale=1.0)
```

Ground-truth pocket labels are generated using VolSite: VolSite (integrated in the IChem toolkit) can generate corresponding binding cavities based on ligands, stored in mol2 format.[^*]

[^*]: Desaphy J, Azdimousa K, Kellenberger E, et al. *Comparison and druggability prediction of protein–ligand binding sites from pharmacophore-annotated cavity shapes*. 2012.


---

## Model

You can download our pre-trained model from the following link:
* **Download Link**: [Download](https://pan.baidu.com/s/1V99HXwIoMHBU3NY9FpbA7w)
* **Password**: `a3rv`

---

## Usage
 
### Training
```bash
python train.py -b 5 -o SGLEPocket/model/ -d scPDB --train_types seg_scPDB_train0.types --test_types seg_scPDB_test0.types
```

*The complete training code will be released upon paper acceptance.*

### Testing

#### 1. DCC Evaluation
```bash
python test.py --test_set coach420 --model_path bestmodel.pth.tar --DATA_ROOT /dataset --is_dca 0
```

#### 2. DCA Evaluation
To perform DCA evaluation with Top-N ranking (where `n` represents the number of real pockets used to control the prediction output):

**Top-n:**
```bash
python test.py --test_set coach420 --model_path bestmodel.pth.tar --DATA_ROOT /dataset --is_dca 1 -n 0
```

**Top-n+2:**
```bash
python test.py --test_set coach420 --model_path bestmodel.pth.tar --DATA_ROOT /dataset --is_dca 1 -n 2
```
