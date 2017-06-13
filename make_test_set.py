import pickle as pkl
import shutil
import os

rand_urls = pkl.load(open("random_urls_50000.p","rb"))
dumb_map = {}
print rand_urls[0]
new_urls = []
for urls in rand_urls: 
	last_pt = urls.split('/')[-1].strip() 
	result = last_pt.split('_')[1:-1]
	s = '_'.join(result) + '.png'
	dumb_map[s] = urls
	new_urls.append(s) 
urls_set = set(new_urls) 
print "rand map set", len(urls_set) 


# print rand_urls

all_files = []

low_files = os.listdir('./images_50k/train/low/')
med_files = os.listdir('./images_50k/train/med/')
high_files = os.listdir('./images_50k/train/high/')
high_val_files = os.listdir('./images_50k/val/high/')
med_val_files = os.listdir('./images_50k/val/med/')
low_val_files = os.listdir('./images_50k/val/low/')

all_files = low_files + med_files + high_files + low_val_files + med_val_files + high_val_files
# print len (all_files) 
print len(all_files)
print all_files[0]
all_files = set(all_files) 


remainder = urls_set - all_files 
print len(remainder)

url_list = []
count = 0 
for r in remainder: 
	count += 1
	url_list.append(dumb_map[r])
	if count == 4000: 
		break 

# print url_list

print len(url_list)

# counter = 0 
# countertru = 0 

# # 37.879689_-122.259415_240.000000.png
# # f = '37.879689_-122.259415_240.000000.png'
# for f in all_files:
# 	if f == 'test': 
# 		continue
# 	lat = f.split('_')[0].strip() 
# 	long = f.split('_')[1].strip()
# 	angle = f.split('_')[2].strip()
# 	present = False 
# 	for url in rand_urls:
# 		counter += 1
# 		if lat in url and long in url and angle in url: 
# 			# print "HERE"
# 			countertru += 1 
# 			present = True
# 			# break
# 			# print "THIS SHOULD NEVER BE PRINTED"
# 	# print present 
# 	if not present: 
# 		new_urls.add(url)
# 	if len(new_urls) == 1000: 
# 		break 


# print "new urls: ", len(new_urls)
pkl.dump(url_list, open('test_urls.p', 'w+'))
# print counter, countertru 
