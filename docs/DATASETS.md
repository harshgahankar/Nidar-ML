# Datasets

## Sources (all manual-download — licensing/auth required, never auto-scraped)

| Source | Access | Target layout |
|---|---|---|
| VisDrone-DET | https://github.com/VisDrone/VisDrone-Dataset — accept terms, download `VisDrone2019-DET-{train,val,test}` zips | `datasets/raw/visdrone/VisDrone2019-DET-train/{images,annotations}` |
| SARD | `kaggle datasets download nikolasgegenava/sard-search-and-rescue -p datasets/raw/sard --unzip` (needs Kaggle API token) | `datasets/raw/sard/...` |
| Mannequin | https://universe.roboflow.com/drone-e2zlz/mannequin-detection-bzihs — export YOLOv8/YOLO11 format, unzip | `datasets/raw/mannequin/{images,labels,data.yaml}` |
| Custom 20 m | your own flights | `datasets/raw/custom/<session>/*.{mp4,jpg}` + `flight_meta.csv` (altitude, camera) |

## Label policy

- Single class `0 = survivor` (standing/walking/sitting/lying humans + mannequins + partial figures).
- Future `human(0)/mannequin(1)` split is reserved in `configs/dataset.yaml` — do not use until relabelled.
- Unlabelled survivors must NOT be silently treated as background: `sard_importer` stages
  unlabelled images separately and blocks them from training until annotated.
- Corrupt images, invalid boxes, exact duplicates are removed; near-duplicates grouped by dHash.

## Pipeline

```
python scripts/prepare_dataset.py [--skip-visdrone --skip-sard --skip-mannequin --skip-custom]
# -> datasets/staging/<source>/<session>/{*.jpg,*.txt}
# -> datasets/processed/{images,labels}/{train,val,test}/ + data.yaml
# -> datasets/reports/{validation_*.json, dataset_stats.json}
```

Split: 70/15/15 grouped by flight/session (`group_by_session: true`) to prevent leakage.
