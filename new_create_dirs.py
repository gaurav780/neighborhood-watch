import pickle as pkl
import shutil 
import os 

lat_longs =pkl.load(open("lat_longs_10000.p","rb"))
ll_to_buckets = pkl.load(open("ll_to_buckets_10000.p","rb"))

files = os.listdir('./images_10000')
cutoff = int(len(files)*0.8) 
idx = 0 

for file in files: 
	print "Processing " + file 
	f_split = file.split("_")
	ll = (f_split[0], f_split[1])
	# print (ll) 
	if ll in ll_to_buckets:
		label = ll_to_buckets[ll]
	else:
		continue
	file = './images_10000/' + file 
	# if idx == 20: 
	# 	break 
	if idx < cutoff: 
		if (label == 0): 
			shutil.move(file, 'train/low/')
		elif (label == 1): 
			shutil.move(file, 'train/med/')
		else: 
			shutil.move(file, 'train/high') 
	else: 
		if (label == 0): 
			shutil.move(file, 'val/low/')
		elif (label == 1): 
			shutil.move(file, 'val/med/')
		else: 
			shutil.move(file, 'val/high') 
	idx += 1

# for ll in lat_longs[:3]: 
# 	print ll

# print ll_to_buckets

# for ll in lat_longs[:800]:
# 	label = ll_to_buckets[ll]
# 	if (label == 0): 
# 		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/low/')
# 	elif (label == 1): 
# 		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/med/')
# 	else: 
# 		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/high/')

# for ll in lat_longs[800:]:
# 	label = ll_to_buckets[ll]
# 	if (label == 0): 
# 		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/low/')
# 	elif (label == 1): 
# 		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/med/')
# 	else: 
# 		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/high/')
