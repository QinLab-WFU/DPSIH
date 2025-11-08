import os
import re

import scipy.io as scio
import numpy as np

# mkdir mat
# mv make_nuswide.py mat
# python make_nuswide.py
# 主要功能是将NUS-WIDE数据集处理成MATLAB可以读取的格式

# 数据集根目录
root_dir = "/home/user/Projects/dataset/nuswide"
# 图像列表文件路径
imageListFile = os.path.join(root_dir, "ImageList", "Imagelist.txt")
# 标签文件夹路径
labelPath = os.path.join(root_dir, "Groundtruth", "AllLabels")
# 标题文件路径
textFile = os.path.join(root_dir, "NUS_WID_Tags", "All_Tags.txt")
# 类别索引文件路径
classIndexFile = os.path.join(root_dir, "ConceptsList", "Concepts81.txt")

# 图像路径
# you can use the image urls to download images
imagePath = os.path.join("/home/user/Projects/dataset/nuswide/images")

# 读取图像列表文件，获取图像路径
with open(imageListFile, "r") as f:
    indexs = f.readlines()

# 提取每个路径的文件名部分，并与 imagePath 拼接
indexs = [os.path.join(imagePath, item.strip().split("\\")[-1]) for item in indexs]
print("indexs length:", len(indexs))


#class_index = {}
#with open(classIndexFile, "r") as f:
#    data = f.readlines()
#
#for i, item in enumerate(data):
#    class_index.update({item.strip(): i})

captions = []
with open(textFile, "r", encoding='utf-8') as f:
    for line in f:
        if len(line.strip()) == 0:
            print("some line empty!")
            continue
        caption = line.split()[1:]
        caption = " ".join(caption).strip()
        # caption = re.sub(r'[^a-zA-Z]+', "", str(caption))
        if len(caption) == 0:
             caption = "123456"
        captions.append(caption)

print("captions length:", len(captions))


with open("/home/user/Projects/dataset/nuswide/Groundtruth/used_label.txt", encoding='utf-8') as f:
    label_lists = f.readlines()
label_lists = [item.strip() for item in label_lists]

class_index = {}
for i, item in enumerate(label_lists):
    class_index.update({item: i})

labels = np.zeros([len(indexs), len(class_index)], dtype=np.int8)

for item in label_lists:
    path = os.path.join(labelPath, item)
    class_label = item# .split(".")[0].split("_")[-1]

    with open(path, "r") as f:
        data = f.readlines()
    for i, val in enumerate(data):
        labels[i][class_index[class_label]] = 1 if val.strip() == "1" else 0
print("labels sum:", labels.sum())

not_used_id = []
with open("/home/user/Projects/dataset/nuswide/Groundtruth/not_used_id.txt", encoding='utf-8') as f:
    not_used_id = f.readlines()
not_used_id = [int(int(item.strip())-2) for item in not_used_id]

# for item in not_used_id:
#     indexs.pop(item)
#     captions.pop(item)
#     labels = np.delete(labels, item, 0)
ind = list(range(len(indexs)))
for item in not_used_id:
    ind.remove(item)
    indexs[item] = ""
    captions[item] = ""
indexs = [item for item in indexs if item != ""]
captions = [item for item in captions if item != ""]
ind = np.asarray(ind)
labels = labels[ind]
# ind = range(len(indexs))

print("indexs length:", len(indexs))
print("captions length:", len(captions))
print("labels shape:", labels.shape)

indexs = {"index": indexs}
captions = {"caption": captions}
labels = {"category": labels}

scio.savemat('/home/user/Projects/dataset/nuswide/index.mat', indexs)
# scio.savemat("caption.mat", captions)
scio.savemat('/home/user/Projects/dataset/nuswide/label.mat', labels)

# 读取标签文件，构建标签矩阵
captions = [item + "\n" for item in captions["caption"]]

with open('/home/user/Projects/dataset/nuswide/caption.txt', "w", encoding='utf-8') as f:
    f.writelines(captions)

print("finished!")

