import requests
import zipfile
import io
import os
import pandas as pd

url = "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/glossary-dataset-indoword.zip"
print(f"Downloading {url}...")
try:
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            print("Zip contents:", z.namelist()[:10])
            for filename in z.namelist():
                if any(k in filename.lower() for k in ['kok', 'konkani', 'gom', 'goan']):
                    print("FOUND KONKANI FILE:", filename)
                    # extract and load
                    z.extract(filename, "data/raw/ai4bharat/")
                    fp = os.path.join("data/raw/ai4bharat/", filename)
                    df = pd.read_csv(fp, sep=None, engine='python')
                    print("Loaded shape:", df.shape)
                    print(df.head(3))
except Exception as e:
    print("Error:", e)
