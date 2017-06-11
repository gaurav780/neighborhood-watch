from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

print "starting"
map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,
    resolution = 'h', area_thresh = 0.1,
    llcrnrlon=-122.4, llcrnrlat=37.7,
    urcrnrlon=-122.15, urcrnrlat=37.9)
 
map.drawcoastlines()
map.drawmapboundary(fill_color='#ADD8E6')
map.drawcountries()
map.fillcontinents(color = '#C0C0C0')
map.drawmapboundary()

# Plot ground truth mappings
fname = "misclassified-50k.txt"

correct = []
corr_low = []
corr_med = []
corr_high = []
incorrect = []
with open(fname) as f:
	for currline in f:
		line = currline.strip().split()
		if line[1] == line[2]:
			correct.append(line)
			if line[1].strip() == '1':
				corr_low.append(line)
			elif line[1].strip() == '2':
				corr_med.append(line)
			else:
				corr_high.append(line)
		else:
			incorrect.append(line)		

##### THIS SECTION FOR DISPLAYING CORRECT/INCOREECT PRED
lat_corr = []
long_corr = []
for corr in incorrect:
        coord = corr[0].split('_')
        lat_corr.append(float(coord[1].split('/')[-1]))
        long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.scatter(x, y,c='#ff0000', marker='o', s=1,zorder=10)
lat_corr = []
long_corr = []
for corr in correct:
        coord = corr[0].split('_')
        lat_corr.append(float(coord[1].split('/')[-1]))
        long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.scatter(x, y, c='#9370db',marker='o', s=1,zorder=10)
'''
##### THIS SECTION FOR DISPLAYING TRUE SECTION #######
lat_corr = []
long_corr = []
for corr in corr_low:
	coord = corr[0].split('_')
	lat_corr.append(float(coord[1].split('/')[-1]))
	long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.scatter(x, y,c='#ee82ee', marker='o', s=1,zorder=10)
lat_corr = []
long_corr = []
for corr in corr_med:
        coord = corr[0].split('_')
        lat_corr.append(float(coord[1].split('/')[-1]))
        long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.scatter(x, y, c='#ba55d3',marker='o', s=1,zorder=10)
lat_corr = []
long_corr = []
for corr in corr_high:
        coord = corr[0].split('_')
        lat_corr.append(float(coord[1].split('/')[-1]))
        long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.scatter(x, y,c='#800080', marker='o', s=1,zorder=10)
'''
print "plotted markers?"
plt.show()
