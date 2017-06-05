import os 
from PIL import Image
from PIL import ImageFile

low_files = os.listdir('./images_50k/train/low/')
med_files = os.listdir('./images_50k/train/med/')
high_files = os.listdir('./images_50k/train/high/')

fail_count = 0 
for file in low_files: 
	try: 
		Image.open('images_50k/train/low/' + file).convert('RGB')
	except: 
		print 'Failed ' + file
		fail_count += 1 

print "train/low"
print fail_count 

fail_count = 0 
for file in med_files: 
	try: 
		Image.open('images_50k/train/med/' + file).convert('RGB')
	except: 
		print 'Failed ' + file
		fail_count += 1 

print "train/med"
print fail_count 

fail_count = 0 
for file in high_files: 
	try: 
		Image.open('images_50k/train/high/' + file).convert('RGB')
	except: 
		print 'Failed ' + file
		fail_count += 1 

print "train/high"
print fail_count 

low_files = os.listdir('./images_50k/val/low/')
med_files = os.listdir('./images_50k/val/med/')
high_files = os.listdir('./images_50k/val/high/')

fail_count = 0 

for file in low_files: 
	try: 
		Image.open('images_50k/val/low/' + file).convert('RGB')
	except: 
		print 'Failed ' + file
		fail_count += 1 

print "val/low"
print fail_count 

fail_count = 0 
for file in med_files: 
	try: 
		Image.open('images_50k/val/med/' + file).convert('RGB')
	except: 
		print 'Failed ' + file
		fail_count += 1 

print "val/med"
print fail_count 

fail_count = 0 
for file in high_files: 
	try: 
		Image.open('images_50k/val/high/' + file).convert('RGB')
	except: 
		print 'Failed ' + file
		fail_count += 1 

print "val/high"
print fail_count 

