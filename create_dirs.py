import pickle as pkl
import shutil 
import os 


# lat_longs =pkl.load(open("lat_longs_1000.p","rb"))
ll_to_buckets = pkl.load(open("ll_to_bucket_50k_fullcorpus.p","rb"))


# for ll in lat_longs[:8000]:
# 	label = ll_to_buckets[ll]
# 	if (label == 0): 
# 		shutil.move('images_1000/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/low/')
# 	elif (label == 1): 
# 		shutil.move('images_1000/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/med/')
# 	else: 
# 		shutil.move('images_1000/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/high/')

# for ll in lat_longs[8000:]:
# 	label = ll_to_buckets[ll]
# 	if (label == 0): 
# 		shutil.move('images_1000/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/low/')
# 	elif (label == 1): 
# 		shutil.move('images_1000/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/med/')
# 	else: 
# 		shutil.move('images_1000/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/high/')

files = os.listdir('./images_1000/')
cutoff = int(len(files)*0.8) 
idx = 0 

for file in files: 
	if file == 'val' or file == 'train' or file == 'med': 
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
	file = './images_1000/' + file 
	if idx < cutoff: 
		if (label == 0): 
			shutil.move(file, './images_1000/train/low/')
		elif (label == 1): 
			shutil.move(file, './images_1000/train/med/')
		else: 
			shutil.move(file, './images_1000/train/high/') 
	else: 
		if (label == 0): 
			shutil.move(file, './images_1000/val/low/')
		elif (label == 1): 
			shutil.move(file, './images_1000/val/med/')
		else: 
			shutil.move(file, './images_1000/val/high/') 
	idx += 1
