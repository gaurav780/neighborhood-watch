# Open file
import sys
import urllib2
import numpy as np
import json
import pickle as pkl

ll_fips = {}
lat_longs = pkl.load(open("lat_longs_1000.p","rb"))
count = 0
inc_file = open("../ACS_15_5YR_S1902_with_ann.csv","rb")
geoid_to_meaninc = {}
for line in inc_file:
  data = line.split(',')
  geoid = data[1]
  mean_inc = data[7]
  geoid_to_meaninc[geoid] = mean_inc
for latlong in lat_longs:
  print (count)
  count+=1
  url = 'http://data.fcc.gov/api/block/find?format=json&latitude=' + latlong[0]+"&longitude="+latlong[1]+'&showall=true'
  try:
    response = urllib2.urlopen(url)
    content = response.read()
    parsed_content = json.loads(content)
    geoid =  parsed_content['Block']['FIPS'][:11]
    mean_i = geoid_to_meaninc[geoid]
    ll_fips[latlong]=mean_i
  except urllib2.URLError as e:
    print(type(e))
pkl.dump(ll_fips,open("ll_to_meaninc_1000.p","w+"))
