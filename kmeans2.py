'''
FOR only 2 Attributes except for ID which is removed and 
then performing clustering 
'''
import numpy as np
import csv
import random
import math

def euclid_dist(data, centroids):
	dist=[]
	for c in centroids:
		row=[]
		for p in data:
			row.append(euclid_value(c,p))
		dist.append(row)
	return dist

def euclid_value(c,p):
	res=0
	for i in range(len(p)):
		# print("{0} {1}".format(p[i],c[i]))
		res+=(int(p[i])-c[i])**2
		# print("{0} {1}".format(type(p[i]),type(c[i])))
		# print(res)
		# print('---')
	res=math.sqrt(res)
	return res

# Reading csv bank dataset
fhand = open('bank_data.csv', 'r')
reader = csv.reader(fhand)
raw_data=[row for row in reader] # reading file as list


# Preprocessing on the dataset to make it suitable for clustering
pre_process = {'MALE':1,'FEMALE':0,'YES':1,'NO':0,'INNER_CITY':0,'TOWN':1,'RURAL':2,'SUBURBAN':3}
data=np.array(raw_data) # convertinfg to numpy array
attributes=data[0] # retirve column names
data=np.delete(data,(0),axis=0) # remove first row of column names

attributes=np.delete(attributes,(0),axis=0) # remove id 
data=np.delete(data,(0),axis=1) # first column of id

#selecting attributes => age, income
print('\nChoose 2 Attributes for clustering\nAttribute => Code')
for i in range(len(attributes)):
	print("{0} => {1}".format(attributes[i],i))
attr1=int(input('Enter Code for Attribute 1 : '))
attr2=int(input('Enter Code for Attribute 2 : '))
if not (attr1 < len(attributes) and attr2 < len(attributes) and attr1 != attr2):
	print('Bad input!!')
	exit(0)
remover=[]
for i in range(len(attributes)):
	if i == attr1 or i == attr2:
		continue
	else:
		remover.append(i)
attributes=np.delete(attributes,tuple(remover),axis=0)
data=np.delete(data,tuple(remover),axis=1)
print(attributes)

# attributes=np.delete(attributes,(0,2,3,5,6,7,8,9,10,11),axis=0)
# data=np.delete(data,(0,2,3,5,6,7,8,9,10,11),axis=1)
data=data.tolist()
# converting textual data to integers
for i in range(len(data)):
	for j in range(len(data[i])):
		if data[i][j] in pre_process:
			data[i][j]=pre_process[data[i][j]]
		else:
			data[i][j]=float(data[i][j])
			data[i][j]=int(data[i][j])

# data=[[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]
data=np.array(data)
data_copy=np.array(data)

# number of clusters required
num_of_clusters=int(input('Enter Number of Clusters (max 5) : '))
if num_of_clusters > 5:
	print('Clusters max 5 allowed => auto assigning num_of_clusters=5')
	num_of_clusters=5
# num_of_clusters=5
maxnum_of_iterations=200
# assigning random centroids / can also use random function below 
#centroids_pos=[0,3,6]
centroids_pos=[183, 372, 201, 549, 8]
distance=[]
clusters=[]
clusters_copy=[]
divide_cluster={}
centroids=[]
for i in range(num_of_clusters):
 	centroids.append(data[centroids_pos[i]].tolist())

print(data)
iterations=0
while iterations < maxnum_of_iterations:
	print('\n\nIteration {0}: -------------------'.format(iterations))
	iterations+=1
	# dist matrix of 600 x 5
	distance=np.array(euclid_dist(data, centroids))
	# print(distance)
	clusters=[]
	divide_cluster={}

	cols=distance.shape[1]
	for i in range(cols):
		temp=np.argmin(distance[:,i].tolist())
		clusters.append(temp)
		if temp in divide_cluster:
			divide_cluster[temp].append(i)
		else:
			divide_cluster[temp]=[i]
		# print(distance[:,i].tolist())
	# print(clusters)
	for i in range(num_of_clusters):
		print('Cluster {0} : '.format(i),end="")
		print(len(divide_cluster[i]))
	# print(data)
	centroids=[]
	for i in range(num_of_clusters):
		temp=divide_cluster[i]
		res=[]
		for j in range(len(temp)):
			if j == 0:
				res=data[temp[j]]
			else:
				for k in range(len(data[temp[j]])):
					res[k]+=data[temp[j]][k]
		res=[p/len(temp) for p in res]
		centroids.append(res)
	# print(centroids)
	data=np.array(data_copy)
	# print(data)
	if np.array_equal(np.array(clusters),np.array(clusters_copy)):
		break
	else:
		clusters_copy=clusters

print('DONE!!!')
print(attributes)
# Plotting all points
import matplotlib.pyplot as plt
import pylab as pl
x = data[:,0]
y = data[:,1]
plt.scatter(x, y)
plt.show()

# Plotting clusters
cplot=[]
for i in range(num_of_clusters):
	cplot.append(list())
for i in range(len(clusters)):
	for j in range(num_of_clusters):
		if clusters[i] == j:
			cplot[j].append(data[i])
for i in range(len(cplot)):
	cplot[i]=np.array(cplot[i])
# c0,c1,c2,c3,c4=[],[],[],[],[]
# for i in range(len(clusters)):
# 	if clusters[i] == 0:
# 		c0.append(data[i])
# 	elif clusters[i] == 1:
# 		c1.append(data[i])
# 	elif clusters[i] == 2:
# 		c2.append(data[i])
# 	elif clusters[i] == 3:
# 		c3.append(data[i])
# 	elif clusters[i] == 4:
# 		c4.append(data[i])
# c0,c1,c2,c3,c4=np.array(c0),np.array(c1),np.array(c2),np.array(c3),np.array(c4)
plotter=[]
colors=['b','g','r','c','m']
markers=['*','+','o','*','+']
for i in range(len(cplot)):
	plotter.append(pl.scatter(cplot[i][:,0],cplot[i][:,1],c=colors[i],marker=markers[i]))

# plot0=pl.scatter(c0[:,0],c0[:,1],c='b',marker='*')
# plot1=pl.scatter(c1[:,0],c1[:,1],c='g',marker='+')
# plot2=pl.scatter(c2[:,0],c2[:,1],c='r',marker='o')
# plot3=pl.scatter(c3[:,0],c3[:,1],c='c',marker='*')
# plot4=pl.scatter(c4[:,0],c4[:,1],c='m',marker='+')

# pl.legend([plot0,plot1,plot2,plot3,plot4], ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
pl.legend(plotter, ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
pl.title('Bank dataset with {2} clusters for {0} and {1}'.format(attributes[0],attributes[1],num_of_clusters))
pl.show()