import requests
import zipfile
import io
import os
import pandas as pd

urls = [
    "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/glossary-dataset-indoword.zip",
    "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/glossary-dataset-bharatvani.zip",
    "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/glossary-dataset-cstt.zip",
    "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/glossary-dataset-osf.zip",
    "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/glossary-dataset-nlpc.zip",
    "https://anuvaad-raw-datasets.s3-us-west-2.amazonaws.com/glossary-dataset-anuvaad.zip"
]

all_dfs = []

for url in urls:
    print(f"Checking {url}...")
    try:
        r = requests.get(url, stream=True, timeout=15)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                print(f"File count in zip: {len(z.namelist())}")
                for filename in z.namelist():
                    f_lower = filename.lower()
                    if any(k in f_lower for k in ['kok', 'konkani', 'gom', 'goan']):
                        print(f"--> FOUND MATCH: {filename}")
                        z.extract(filename, "data/raw/ai4bharat/")
                        fp = os.path.join("data/raw/ai4bharat/", filename)
                        try:
                            df = pd.read_csv(fp, sep=None, engine='python', on_bad_lines='skip')
                            all_dfs.append(df)
                        except Exception as ex:
                            print(f"Read error: {ex}")
    except Exception as e:
        print(f"Failed {url}: {e}")

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    print("Combined Glossary Shape:", combined.shape)
    print(combined.head(5))
