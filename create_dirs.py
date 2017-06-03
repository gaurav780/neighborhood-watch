import pickle as pkl
import shutil 

lat_longs =pkl.load(open("lat_longs_1000.p","rb"))
ll_to_buckets = pkl.load(open("ll_to_buckets_1000.p","rb"))

for ll in lat_longs[:800]:
	label = ll_to_buckets[ll]
	if (label == 0): 
		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/low/')
	elif (label == 1): 
		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/med/')
	else: 
		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'train/high/')

for ll in lat_longs[800:]:
	label = ll_to_buckets[ll]
	if (label == 0): 
		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/low/')
	elif (label == 1): 
		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/med/')
	else: 
		shutil.move('images_test/'+ll[0]+'_'+ll[1]+'_60.000000.png', 'val/high/')
