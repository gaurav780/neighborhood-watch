# Open file
from multiprocessing.pool import ThreadPool
import sys
import urllib2
import numpy as np
import pickle as pkl
import threading
import contextlib
#import concurrent.futures
data_file = '../cityid_url-3.txt'
output_dir = 'images_50000/'
num_images = 0
'''
urls = []

with open(data_file) as f:
	for line in f:
		info = line.split()
		if int(info[0]) == 270:
			urls.append(info[1])
			num_images += 1


indices = np.arange(num_images)
urls = np.array(urls)
#Adjust second parameter for number of images you want to download
rand_ind = np.random.choice(indices, 50000, replace=False)
rand_urls = urls[rand_ind]
pkl.dump(rand_urls, open('random_urls_50000.p', 'w+'))

'''
rand_urls = pkl.load(open('random_urls_50000.p','rb'))



#for url in rand_urls:
def fetch_url(url):
	try: 
		rel = url.split('im_')
		rel = rel[1].split('_0.000000')
		with contextlib.closing(urllib2.urlopen(url, timeout=60)) as response:
		#response = urllib2.urlopen(url, timeout=60)
			content = response.read()
			f_dest = open(output_dir + rel[0] + '.png', 'w+')
			print (output_dir + rel[0] + '.png')
			f_dest.write(content)
			print("Found")
			f_dest.close()
	except urllib2.URLError as e:
		print (e.reason)

'''
pool = ThreadPool(1000)
results = pool.imap_unordered(fetch_url, rand_urls[:5000])
pool.close()
pool.join()

'''
threads = [threading.Thread(target=fetch_url, args=(url,)) for url in rand_urls[:1000]]
for thread in threads:
  thread.start()
for thread in threads:
  thread.join()	
