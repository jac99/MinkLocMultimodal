import pickle

with open("lidar2image_ndx.pickle", "rb") as f:
    data = pickle.load(f)

print('.')

with open("/media/sf_Datasets/pointnetvlad/benchmark_datasets/test_queries_baseline.pickle", "rb") as f:
    data = pickle.load(f)

temp = [data[e].timestamp for e in data if data[e].timestamp==1400505900453981]
print(temp)
print('.')
