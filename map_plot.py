from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

print "starting"
map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,
    resolution = 'h', area_thresh = 0.1,
    llcrnrlon=-122.5, llcrnrlat=37.5,
    urcrnrlon=-122.0, urcrnrlat=38.0)
 
map.drawcoastlines()
map.drawmapboundary(fill_color='#ADD8E6')
map.drawcountries()
map.fillcontinents(color = '#C0C0C0')
map.drawmapboundary()

# Plot ground truth mappings
fname = "misclassified.txt"

corr_low = []
corr_med = []
corr_high = []
incorrect = []
with open(fname) as f:
	for currline in f:
		line = currline.strip().split()
		if line[1] == line[2]:
			if line[1].strip() == '0':
				corr_low.append(line)
			elif line[1].strip() == '1':
				corr_med.append(line)
			else:
				corr_high.append(line)
		else:
			incorrect.append(line)		

lat_corr = []
long_corr = []
for corr in corr_low:
	coord = corr[0].split('_')
	lat_corr.append(float(coord[1].split('/')[-1]))
	long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.plot(x, y, 'bo', markersize=5)
lat_corr = []
long_corr = []
for corr in corr_med:
        coord = corr[0].split('_')
        lat_corr.append(float(coord[1].split('/')[-1]))
        long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.plot(x, y, 'go', markersize=5)
lat_corr = []
long_corr = []
for corr in corr_high:
        coord = corr[0].split('_')
        lat_corr.append(float(coord[1].split('/')[-1]))
        long_corr.append(float(coord[2]))
x, y = map(long_corr, lat_corr)
map.plot(x, y, 'yo', markersize=5)
print "plotted markers?"
plt.show()
