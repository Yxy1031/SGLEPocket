# Single Protein Pocket Prediction 

Main workflow: Protein Cleaning → FPocket Candidate Pocket Identification → Candidate Center Extraction → Data Format Conversion → Pocket Ranking → Pocket Segmentation Prediction

In this demo, we use `2qeh.pdb` as an example to demonstrate how to perform pocket prediction using SGLEPocket.

## Steps 1-5 are the same as the DeepPocket processing pipeline

## Step 1: Clean PDB File

### Purpose
Remove water molecules, ligands, metal ions, and other impurities from the PDB file, keeping only standard amino acid residues (protein backbone and side chains).

```bash
python clean_pdb.py <input.pdb> <output_nowat.pdb>
```

### Example
```bash
python clean_pdb.py 2qeh.pdb 2qeh_nowat.pdb
```

### Description
- **Input**: Original PDB file (e.g., `2qeh.pdb`)
- **Output**: Cleaned PDB file (e.g., `2qeh_nowat.pdb`)

---

## Step 2: Generate Candidate Pockets Using fpocket

### Purpose
Use fpocket software to identify potential binding pockets in the protein.

```bash
fpocket -f <input_nowat.pdb>
```

### Example Command
```bash
fpocket -f 2qeh_nowat.pdb
```

### Description
- **Input**: Cleaned PDB file
- **Output**: fpocket output directory (e.g., `2qeh_nowat_out/`) containing candidate pocket information
- **Output Structure**: `<basename>_out/pockets/` directory contains detailed information for each pocket

---

## Step 3: Extract Candidate Pocket Center Coordinates

### Purpose
Extract the barycenter coordinates of each candidate pocket from fpocket output files.

### General Command
```bash
python get_centers.py <fpocket_output_dir>/pockets
```

### Example Command
```bash
python get_centers.py 2qeh_nowat_out/pockets
```

### Description
- **Input**: pockets directory from fpocket output
- **Output**: `bary_centers.txt` file containing center coordinates of all candidate pockets
- **Output Location**: `<fpocket_output_dir>/pockets/bary_centers.txt`

---

## Step 4: Data Format Conversion, Generate Binary Atomic Feature List (.gninatypes) and Corresponding Center Coordinates and Path Index File (.types)

### Purpose
Convert protein structure and candidate center coordinates to the format required by the DeepPocket model.

### General Command
```bash
python types_and_gninatyper.py <protein_nowat.pdb> <fpocket_output_dir>/pockets/bary_centers.txt
```

### Example Command
```bash
python types_and_gninatyper.py 2qeh_nowat.pdb 2qeh_nowat_out/pockets/bary_centers.txt
```

### Description
- **Input**:
  - Cleaned PDB file
  - Candidate center coordinate file `bary_centers.txt`
- **Output**:
  - `protein_clean.gninatypes`
  - `pockets/bary_centers.types`

---

## Step 5: Rank Candidate Pockets Using DeepPocket Model

### Purpose
Use a pre-trained DeepPocket model to score and rank candidate pockets.

**Download the corresponding model file from the DeepPocket project: [DeepPocket](https://github.com/devalab/DeepPocket)**

### General Command
```bash
python rank_pockets.py \
  --model <model.py> \
  --test_types <fpocket_output_dir>/pockets/bary_centers.types \
  --checkpoint <pretrained_model.pth.tar>
```

### Example Command
```bash
python rank_pockets.py \
  --model model.py \
  --test_types 2qeh_nowat_out/pockets/bary_centers.types \
  --checkpoint best_test_auc_85001.pth.tar
```

### Description
- **Input**:
  - Model definition file `model.py`
  - Candidate center type file `bary_centers.types`
  - Pre-trained ranking model weight file `.pth.tar`
- **Output**: `bary_centers_ranked.types` (ranked list of candidate pockets)
- **Output Location**: `<fpocket_output_dir>/pockets/bary_centers_ranked.types`

---

## Step 6: Perform Precise Pocket Shape Segmentation Prediction

### Purpose
Use a trained segmentation model to perform fine-grained segmentation on ranked candidate pockets and predict pocket regions.

### General Command
```bash
python segsingle.py \
  --model_path <checkpoint.pth.tar> \
  --types_file <fpocket_output_dir>/pockets/bary_centers_ranked.types \
  --data_dir <working_directory> \
  --output_dir <output_directory> \
  --threshold <threshold_value> \
  --gpu <gpu_id>
```

### Example Command
```bash
python segsingle.py \
  --model_path bestmodel.pth.tar \
  --types_file 2qeh_nowat_out/pockets/bary_centers_ranked.types \
  --data_dir ./ \
  --output_dir ./predresults \
  --threshold 0.5 \
  --gpu 0
```

### Description
- **Input**:
  - Model weight file `bestmodel.pth.tar`
  - Ranked candidate pocket file `bary_centers_ranked.types`
  - Working directory (containing protein structure files)
- **Output**: Pocket segmentation prediction result cube
- **Output Location**: `<output_directory>/`

---

## Notes

- Please ensure all dependencies are properly installed before running the scripts.
- Adjust parameters and paths in the scripts according to your actual needs.

---

## Reference & Acknowledgement

The data processing pipeline (Steps 1-5) of this project is based on the core logic and code from the **DeepPocket** framework. If you use these functionalities in your research, please cite the original paper:

**DeepPocket: Ligand Binding Site Detection and Segmentation using 3D Convolutional Neural Networks**  
*Rishal Aggarwal, Akash Gupta, U. Deva Priyakumar*  
Journal of Chemical Information and Modeling (2022)  
**DOI**: [10.1021/acs.jcim.1c00799](https://doi.org/10.1021/acs.jcim.1c00799)  
**GitHub**: [https://github.com/devalab/DeepPocket](https://github.com/devalab/DeepPocket)

If you need BibTeX format:

```bibtex
@article{aggarwal2022deeppocket,
  title={DeepPocket: Ligand Binding Site Detection and Segmentation using 3D Convolutional Neural Networks},
  author={Aggarwal, Rishal and Gupta, Akash and Priyakumar, U Deva},
  journal={Journal of Chemical Information and Modeling},
  volume={62},
  number={21},
  pages={5069--5079},
  year={2022},
  publisher={ACS Publications}
}
```
