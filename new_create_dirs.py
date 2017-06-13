import pickle as pkl
import shutil 
import os 

# lat_longs =pkl.load(open("lat_longs_50000.p","rb"))
# low_cutoff = 1608
# med_cutoff = 2378
# high_cutoff = 2938

# low_files = os.listdir('./train/low/')
# med_files = os.listdir('./train/med/')
# high_files = os.listdir('./train/high/')

# idx = 0 
# # for file in low_files: 
# # 	if idx == low_cutoff: 
# # 		break
# # 	file = './train/low/' + file
# # 	shutil.move(file, './val/low/')
# # 	idx += 1

# # for file in med_files: 
# # 	if idx == med_cutoff: 
# # 		break
# # 	file = './train/med/' + file
# # 	shutil.move(file, './val/med/')
# # 	idx += 1

# for file in high_files: 
# 	if idx == high_cutoff: 
# 		break
# 	file = './train/high/' + file
# 	shutil.move(file, './val/high/')
# 	idx += 1


ll_to_buckets = pkl.load(open("ll_to_bucket_50k_fullcorpus.p","rb"))

files = os.listdir('./test_images/')
# cutoff = int(len(files)*0.8) 
idx = 0 

for file in files: 
	if file == 'high' or file == 'low' or file == 'med': 
		continue 
	print "Processing " + file 
	f_split = file.split("_")
	ll = (f_split[0], f_split[1])
	# print (ll) 
	if ll in ll_to_buckets:
		label = ll_to_buckets[ll]
	else:
		print "Not in map!"
		continue
	file = './test_images/' + file 
	# if idx < cutoff: 
	if (label == 0): 
		shutil.move(file, 'test/low/')
	elif (label == 1): 
		shutil.move(file, 'test/med/')
	else: 
		shutil.move(file, 'test/high/') 
	# else: 
	# 	if (label == 0): 
	# 		shutil.move(file, 'val/low/')
	# 	elif (label == 1): 
	# 		shutil.move(file, 'val/med/')
	# 	else: 
	# 		shutil.move(file, 'val/high/') 
	# idx += 1
