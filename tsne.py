from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

frame_lvl_record = "./train-0.tfrecord"
video_lvl_record = "./video-0.tfrecord"


feat_rgb = []
feat_audio = []

for example in tf.python_io.tf_record_iterator(frame_lvl_record):        
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
    sess = tf.InteractiveSession()
    rgb_frame = []
    audio_frame = []
    # iterate through frames
    for i in range(n_frames):
        rgb_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval())
        audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval())
        
        
    sess.close()
    feat_rgb.append(rgb_frame)
    feat_audio.append(audio_frame)
    break

print('The first video has %d frames' %len(feat_rgb[0]))
print(np.shape(feat_rgb))
print(np.shape(feat_audio))

X = np.zeros([301, 1024])
X[:300] = feat_rgb[0]
X[-1] = np.mean(feat_rgb[0], axis=0) / 300.


X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X) 
fig = plt.figure()
ax = fig.add_subplot(111)
print(np.shape(X_embedded))

cax = ax.scatter(X_embedded[:300, 0], X_embedded[:300, 1], c = np.asarray(range(300)).astype(float), marker="o",edgecolor='none', cmap='inferno')
ax.scatter(X_embedded[-1, 0], X_embedded[-1, 1], c = 1., marker="o",edgecolor='none', cmap='inferno', s=100)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off', 
    right='off',
    labelleft='off', 
    labelbottom='off') # labels along the bottom edge are off
fig.colorbar(cax, ticks=[0, 300.])
plt.savefig('tsne_frame')



vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for example in tf.python_io.tf_record_iterator(video_lvl_record):
    tf_example = tf.train.Example.FromString(example)

    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(tf_example.features.feature['labels'].int64_list.value)
    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
print('Number of videos in this tfrecord: ',len(mean_rgb))
print('First video feature length',len(mean_rgb[0]))
print('First 20 features of the first youtube video (',vid_ids[0],')')

n=10
from collections import Counter
label_mapping = pd.Series.from_csv('./label_names.csv',header=0).to_dict()

top_n = Counter([item for sublist in labels for item in sublist]).most_common(n)
top_n_labels = [int(i[0]) for i in top_n]
top_n_label_names = [label_mapping[x] for x in top_n_labels]
print(top_n_label_names)

colors = plt.cm.rainbow(np.linspace(0, 1, n))
mean_rgb_top_n = []
labels_for_tsne = []
# filtering mean_rgb so it only contains top n labels
for idx, list_of_nodes in enumerate(labels):
    for node in list_of_nodes:
        if node in top_n_labels:
            mean_rgb_top_n.append(mean_rgb[idx])
            labels_for_tsne.append(node)


X_embedded = TSNE(n_components=2, random_state=0).fit_transform(mean_rgb_top_n) 


fig = plt.figure()
ax = fig.add_subplot(111)

handles = []
for indx, color in enumerate(colors):
    this_label = top_n_labels[indx]
    X_embedded_filtered = X_embedded[np.array([x==this_label for x in labels_for_tsne])]
    handles.append(ax.scatter(X_embedded_filtered[:, 0], X_embedded_filtered[:, 1], c=color, marker="o",edgecolor='none'))

ax.legend(handles, top_n_label_names, fontsize=10)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off', 
    right='off',
    labelleft='off', 
    labelbottom='off')
plt.savefig('tsne_video')

plt.show()







