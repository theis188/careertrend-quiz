import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import json
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from collections import defaultdict
import math

def get_content(s):
	s=re.sub(r'<[^>]+>','',s)
	s=s.lower()
	s=re.sub(r'(\n|\t|\r)',' ',s)
	s=re.sub(r' +',' ',s)
	s=re.sub( r'[{}]'.format(punctuation),'', s )
	s=re.sub(r'\d','',s )
	return s

summary_sections = ['summary_what_they_do','summary_work_environment','summary_how_to_become_one','summary_outlook']
sections = ['what_they_do','work_environment','how_to_become_one','job_outlook']
occ_dict=defaultdict(list)
occs = []
tree = ET.parse('xml-compilation.xml')
root = tree.getroot()
# names = [name for name in df['name']]

for occupation in root.findall('occupation'):
	text = ''
	title = occupation.find('title').text
	# if title not in names:
	# 	continue
	text+=title
	occs.append(title)
	for sect in summary_sections:
		try:
			text+=' '+get_content ( occupation.find(sect).text )
		except:
			pass
	for sect in sections:
		occ_dict[sect].append( get_content( occupation.find(sect).find('section_body').text ) )

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')
corpus = occ_dict['what_they_do']
tfidf = tfidf_vectorizer.fit_transform( corpus )
words = tfidf_vectorizer.get_feature_names()
idf = tfidf_vectorizer.idf_
sortr = list( np.argsort(idf) ) [::-1]

def final_records(curr_occs):
	ret = {"type": "result",
		   "results":[]}
	for name in curr_occs:
		ret["results"].append( name )
	return ret

def get_equal_labels_from_centers(centers,arr,n):
	lab_dict = defaultdict(list)
	total = arr.shape[0]
	labels = [0]*total
	max_amt = math.ceil(1.*total/n)
	distances = np.array([ [ euclidean(centers[k,:],arr[i,:]) for k in range(n)] for i in range(total) ])
	order = list( np.argsort( np.min(distances, axis=1) ) )
	for i in order:
		label_order = np.argsort( distances[i,:] )
		for lab in label_order:
			if len(lab_dict[lab])<max_amt:
				lab_dict[lab].append(i)
				labels[i] = lab
				break
	return run_iter(arr,labels)

def generate_proposals(labels,distances):	
	props = {k:[] for k in set(labels)}
	for i in range(distances.shape[0]):
		lab = labels[i]
		lab_order = list( np.argsort( distances[i,:] ) )
		for j in lab_order:
			if j==lab: break
			props[lab].append({'index':i, 'to':j, 'improvement': distances[i,lab]-distances[i,j]}) 
	for lab in props:
		props[lab] = sorted(props[lab], key = lambda dic: dic['improvement'])
	return props

def get_centers(arr,labels):
	ret = np.zeros( (len(set(labels)) , arr.shape[1]) )
	for k in range( ret.shape[0] ):
		indices = [n for n,l in enumerate(labels) if k==l]
		ret[k,:] = np.mean( arr[indices,:] ,axis=0)
	return ret

def find_switch(from_label,pt,props,switch_indices):
	to_label = pt['to']
	for prop in props[to_label]:
		if prop['index'] in switch_indices:
			continue
		if prop['to'] == from_label:
			return prop

def iterate_step(arr,labels):
	total = arr.shape[0]
	old_labels = labels[:]
	centers = get_centers(arr,labels)
	n = centers.shape[0]
	distances = np.array([ [ euclidean(centers[k,:],arr[i,:]) for k in range(n)] for i in range(total) ])
	props = generate_proposals(labels,distances)
	# print(props)
	switch_indices = set([])
	for lab in props:
		print('len props',len(props))
		for pt in props[lab]:
			if pt['index'] in switch_indices:
				continue
			switch_rec = find_switch(lab,pt,props,switch_indices)
			if switch_rec:
				switch_ind = switch_rec['index']
				switch_lab = labels[switch_ind]
				labels[pt['index']] = switch_lab
				labels[switch_ind] = lab
				print('switched',lab,pt['index'],'and',switch_lab,switch_ind,'improvement',pt['improvement']+switch_rec['improvement'])
				switch_indices.update([switch_ind,pt['index']])
	return labels

def run_iter(arr,labels):
	old_labels = labels[:]
	new_labels = iterate_step(arr,labels)
	while new_labels != old_labels:
		old_labels = new_labels[:]
		new_labels = iterate_step(arr,labels)
	return new_labels

def label_equal_wrapper(arr,n=3):
	kmean_obj = KMeans(n_clusters = n)
	km = kmean_obj.fit_transform(arr)
	centers = kmean_obj.cluster_centers_
	return centers,get_equal_labels_from_centers(centers,arr,n)

def generate_record(curr_tfidf,occ_names):
	if len(occ_names) <= 5: return final_records(occ_names)
	record = []
	nmf_solver = NMF(n_components=5, random_state=1,
	          alpha=.01, l1_ratio=1.)
	try:
		nmf = nmf_solver.fit_transform(curr_tfidf)
	except:
		print(curr_tfidf)
		raise Exception
	centers,labels = label_equal_wrapper(nmf,n=3)
	print (labels)
	for k in set( labels ):
		indices = [ind for ind,l in enumerate(labels) if l==k]
		print(indices)
		comp = np.sum( curr_tfidf[indices,:], axis=0 )
		comp = np.array(comp)[0]
		word_order_relevance = [words[i] for i in list(np.argsort( comp ) )[::-1][:100] ]
		words_and_sizes = [{'text':word, 'size':15+35//((k+1)/2) } 
								for k,word in 
								enumerate( word_order_relevance[:25] ) ]
		one_record = {"words":words_and_sizes}
		one_record["type"] = "question"
		indices = [ind for ind,l in enumerate(labels) if l==k]
		new_tfidf = curr_tfidf[indices,:]
		new_occs = [occ_names[i] for i in indices]
		one_record["choice"] = generate_record(new_tfidf, new_occs)
		one_record["num_occs"] = len(new_occs)
		record.append(one_record)
	return record

if __name__ == '__main__':
	this = generate_record(tfidf,occs)

with open('QUIZ_EQUAL_NEW_test.json','w') as f:
	f.write(json.dumps(this))

# if __name__ == '__main__':