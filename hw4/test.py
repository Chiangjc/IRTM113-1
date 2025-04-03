import os
import glob
# import nltk
# from nltk.stem import PorterStemmer
import numpy as np
from collections import defaultdict
import math

import heapq


# nltk.download('punkt')

# 讀取檔案
def load_vector(doc_name):
    vector = {}
    with open(doc_name, 'r', encoding='utf-8') as file:
        next(file)  # The number of terms document has 
        for line in file:
            index, value = line.split()
            vector[int(index)] = float(value)
    return vector

def cosine(docx, docy):
    vector_x = load_vector(docx)
    vector_y = load_vector(docy)
    
    # 所有出現的單字集
    all_indices = set(vector_x.keys()).union(set(vector_y.keys()))
    
    # 統一長度將沒出現的單字填入0
    tf_idf_x = np.array([vector_x.get(term, 0) for term in all_indices])
    tf_idf_y = np.array([vector_y.get(term, 0) for term in all_indices])
    
    # Calculate the cosine similarity
    dot = np.dot(tf_idf_x, tf_idf_y)
    len_x = np.linalg.norm(tf_idf_x)
    len_y = np.linalg.norm(tf_idf_y)
    if len_x == 0 or len_y == 0:
        return 0.0

    cosine_similarity = dot / (len_x * len_y)
    return cosine_similarity

def merge_clusters(cluster1, cluster2):
#     print(merge, cluster1, cluster2)
    return list(set(cluster1 + cluster2))

def update_clusters(clusters, cluster1_index, cluster2_index, merged_cluster):
    updated_clusters = [clusters[i] for i in range(len(clusters)) if i not in [cluster1_index, cluster2_index]]
    updated_clusters.append(merged_cluster)
    return updated_clusters


def calculate_distance(cluster1, cluster2, linkage='single'):
    if linkage == 'single':
        return min(cosine(doc1, doc2) for doc1 in cluster1 for doc2 in cluster2)
    elif linkage == 'complete':
        return max(cosine(doc1, doc2) for doc1 in cluster1 for doc2 in cluster2)
    elif linkage == 'average':
        return np.mean([cosine(doc1, doc2) for doc1 in cluster1 for doc2 in cluster2])
    elif linkage == 'centroid':
        centroid1 = np.mean([load_vector(doc) for doc in cluster1], axis=0)
        centroid2 = np.mean([load_vector(doc) for doc in cluster2], axis=0)
        return np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2))


def HAC(K, heap, linkage='single'):
    

    return clusters
def save_clusters_to_file(clusters, filename):
    with open(filename, 'w') as file:
        for cluster in clusters:
            sorted_cluster = sorted(cluster, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            file.write(' '.join(sorted_cluster) + '\n\n')  # Separate clusters with an empty line


def calculate_initial_distances():
    heap = []
    for i in range(1095):
        print(i+1)
        for j in range(i + 1, 1095):
            distance = cosine(f"output/{i+1}.txt", f"output/{j+1}.txt")
            heapq.heappush(heap, (-distance, (i, j)))  # Use negative because heapq is a min-heap
    return heap

heap_init = calculate_initial_distances()

K = 8
heap = heap_init
clusters = HAC(K, heap, linkage='complete') 
save_clusters_to_file(clusters, f"{K}.txt")
