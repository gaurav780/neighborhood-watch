Steps:
	1. mkdir images_1000
	2.run scrape_oak_1000.py
	3. mkdir train
	4.mkdir val
	5.mkdir train/low, train/med, train/high
	6.mkdir val/low, val/med, val/high
	run create_dirs.py
	mv train images_1000
	mv val images_1000
	then run python tensorflow_finetune.py --train_dir images_1000/train --val_dir images_1000/val
	might have to do this to get 1.2
	sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc0-cp27-none-linux_x86_64.whl
	of tensor flow
