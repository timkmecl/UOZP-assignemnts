import sys
import numpy as np
import glob
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from PIL import Image
from torchvision import transforms


def to_embedding(filename, model):
	"""
	Prejme filename slike in naložen model nevronske mreže, ki jo požene na sliki, vrne output te mreže
	"""
	input_image = Image.open(filename)
	preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	input_tensor = preprocess(input_image)
	input_batch = input_tensor.unsqueeze(0)

	if torch.cuda.is_available():
			input_batch = input_batch.to('cuda')
			model.to('cuda')

	with torch.no_grad():
			output = model(input_batch)

	return output[0].cpu().detach().numpy()

def read_data(path):
	"""
	Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
	kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
	"""

	data = dict()
	model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)

	for filepath in glob.glob(f"{path}/*"):
		filename = os.path.basename(filepath)
		data[filename] = to_embedding(filepath, model)

	return data


def cosine_dist(d1, d2):
	"""
	Vrni razdaljo med vektorjema d1 in d2, ki je smiselno
	pridobljena iz kosinusne podobnosti.
	"""

	dist = 1 -  np.sum(d1*d2) / (np.sqrt(np.sum(np.square(d1))) * np.sqrt(np.sum(np.square(d2))))
	return dist


def k_medoids(data, medoids):
	"""
	Za podane podatke (slovar vektorjev) in medoide vrni končne skupine
	kot seznam seznamov nizov (ključev v slovarju data).
	"""
	count = 0
	change = True
	while change:
		count += 1
		clusters_dict = {m:[] for m in medoids}
		for el in data:
			d_to_ms = [(cosine_dist(data[m], data[el]), m) for m in medoids] # cosin. razdalja od elementa do vseh medoid
			d_to_m = min(d_to_ms)
			clusters_dict[d_to_m[1]].append(el) # el pripada meodidu, ki mu je najbližje
		clusters = [clusters_dict[m] for m in medoids]
		
		new_medoids = []
		change = False
		for m in medoids:
			ds = [(sum(cosine_dist(data[el2], data[el]) for el2 in clusters_dict[m]), el) for el in clusters_dict[m]] # vsote razdalj od vsakega elementa v clustru do vseh drugih
			new_m = min(ds)[1] # nov medoid bo tisti z najmanjšo
			new_medoids.append(new_m)
			if new_m != m:
				change = True
		medoids = new_medoids
	#print(count)
	return clusters


def silhouette(el, clusters, data):
	"""
	Za element el ob podanih podatke (slovar vektorjev) in skupinah
	(seznam seznamov nizov: ključev v slovarju data), vrni silhueto za element el.
	"""
	a = 0
	bs = []
	for cluster in clusters:
		ds = sum(cosine_dist(data[el], data[e]) for e in cluster)/len(cluster) # vsota cosinusnih razdalj do vseh el. v clustru
		if el in cluster:
			if len(cluster) > 1:
				a = ds # če je el. v clustru, je ta vsota a
			else:
				return 0
		else:
			bs.append(ds) # sicer kandidat za b

	b = min(bs) # za b vzame vsoto razdalj do drugega clustra, ki je najmanjša
	s = (b - a)/(max(a, b))
	return s


def silhouette_average(data, clusters):
	"""
	Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
	ključev v slovarju data) vrni povprečno silhueto.
	"""
	s = sum(silhouette(el, clusters, data) for el in data.keys()) # vsota silhuet vseh elementov
	av = s / len(data)
	return av


if __name__ == "__main__":
		if len(sys.argv) == 3:
				K = sys.argv[1]
				path = sys.argv[2]
		else:
				K = 5
				# 3 -> silhouette 0.478
				# 4 -> 0.461
				# 5 -> 0.466
				# 6 -> 0.462
				# 7 -> 0.448
				# 8 -> 0.422
				path = "slike"

		random.seed(42)
		torch.manual_seed(0)

		data = read_data(path)
		

		l = []
		for i in range(0, 100):
			medoids = random.sample(data.keys(), K)
			clusters = k_medoids(data, medoids)
			avg_s = silhouette_average(data, clusters)
			l.append((avg_s, clusters))
		
		# najboljša razporeditev in silhueta od stotih
		clusters = max(l)[1]
		avg_s = max(l)[0]
		print(avg_s)


		for i in range(len(clusters)):
			cluster = clusters[i]
			els = [(silhouette(el, clusters, data), el) for el in cluster]
			els.sort(reverse=True)
			fig, axs = plt.subplots(1, len(els), figsize=(30, 7))

			for j in range(len(els)):
				axs[j].set_title(f'{els[j][0]:.3f}')
				axs[j].axis('off')
				imgpath = os.path.join(path, els[j][1])
				axs[j].imshow(mpimg.imread(imgpath))
			fig.savefig(f'cluster_{i}.png')
			plt.close(fig)

