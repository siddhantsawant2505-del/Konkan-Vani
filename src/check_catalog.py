import os

catalog_dir = "data/raw/ai4bharat/indicnlp_catalog"
for root, dirs, files in os.walk(catalog_dir):
    for f in files:
        fp = os.path.join(root, f)
        print(fp)
