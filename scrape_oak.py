# Open file
import sys
import urllib2
import numpy as np
import wget
import pickle as pkl

data_file = 'cityid_url-3.txt'
output_dir = 'images/'
num_images = 0

'''
urls = []

with open(data_file) as f:
	for line in f:
		info = line.split()
		if int(info[0]) == 270:
			urls.append(info[1])
			num_images += 1


indices = np.arange(1, num_images, 6)
urls = np.array(urls)
#Adjust second parameter for number of images you want to download
rand_ind = np.random.choice(indices, 1000, replace=False)
rand_urls = urls[rand_ind]
pkl.dump(rand_urls, open('random_urls.p', 'w+'))
'''

rand_urls = pkl.load(open('random_urls.p'))


for url in rand_urls:
	try: 
		rel = url.split('im_')
		rel = rel[1].split('_0.000000')

		response = urllib2.urlopen(url)
		content = response.read()
		f_dest = open(output_dir + rel[0] + '.png', 'w+')
		f_dest.write(content)
		f_dest.close()
	except urllib2.URLError as e:
		print type(e)
	break
