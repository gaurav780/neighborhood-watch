import numpy as np
import gpxpy.geo

f = open('misclassified_1000.txt','rb')

pts = []
correct = 0
for line in f:
  line = line.split()
  img_name = line[0]
  true_lab = line[1]
  pred_lab = line[2]
  lat = img_name.split('_')[1].split('/')[-1]
  lng = img_name.split('_')[2]
  pts.append((float(lat),float(lng),pred_lab,true_lab))
  if true_lab == pred_lab:
    correct +=1

print 'initial accuracy:', correct/float(len(pts))
k = 5
new_correct = 0
def getKey(item):
  return item[0]

#haversine distance
for pt in pts:
  lat1 = pt[0]
  lon1 = pt[1]
  lab1 = pt[2]
  true_lab = pt[3]#to verify
  dists_labs = []
  for pt2 in pts:
    if pt == pt2:
      continue
    lat2 = pt2[0]
    lon2 = pt2[1]
    lab2 = pt2[2]
    dist = gpxpy.geo.haversine_distance(lat1, lon1, lat2, lon2)
    dists_labs.append((dist,lab2))
  s = sorted(dists_labs,key=getKey)
  zero_count = 0
  two_count = 0
  one_count = 0
  for p in s[:k]:
    if p[1] == '0':
      zero_count +=1
    elif p[1] == '1':
      one_count +=1
    else:
      two_count +=1
  new_lab = lab1
  if zero_count > two_count and zero_count > one_count:
    new_lab = '0'
  elif one_count > two_count and one_count > zero_count:
    new_lab = '1'
  elif two_count > one_count and two_count > zero_count:
    new_lab = '2'
  if new_lab == true_lab:
    new_correct +=1

print 'new_accuracy',new_correct/float(len(pts))
  
