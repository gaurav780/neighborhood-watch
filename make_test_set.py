import pickle as pkl
import shutil
import os

rand_urls = pkl.load(open("rand_urls_50000.p","rb"))
new_urls = set()

low_files = os.listdir('./train/low/')
med_files = os.listdir('./train/med/')
high_files = os.listdir('./train/high/')
low_val_files = os.listdir('./val/high/')
med_val_files = os.listdir('./val/med/')
low_val_files = os.listdir('./val/med/')

for f in all_files:
  for url in rand_urls:
      continue
    new_urls.add(f)
