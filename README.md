# SMLE

Uruchomienie:
Doker: docker-compose up --build 
Front: python frontend/main.py

Struktura projektu:
/app
├── backend/
│   ├── data/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── annotations/
│   │   │       └── instances_train.json
│   │   └── val/
│   │       ├── images/
│   │       └── annotations/
│   │           └── instances_train.json
│   ├── Mask_RCNN/
│   ├── Faster_RCNN/
│   ├── MRCNN/
│   ├── main.py
│   └── requirements.txt
├── frontend/
├── Dockerfile
├── docker-compose.yml
└── smle_database_dump.sql