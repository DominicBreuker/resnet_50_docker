import os
import numpy as np
from datetime import datetime


def save_results(name, index, extractions, out_dir):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    index = np.array(index)
    extractions = np.array(extractions)

    index_filename = "index_resnet50_{}_{}"\
        .format(name, current_time)
    np.save(os.path.join(out_dir, index_filename), index)
    print("Index shape: {}".format(index.shape))
    print("Load index with numpy: 'np.load('{}.npy')'"
          .format(index_filename))

    extractions_filename = "extractions_resnet50_{}_{}"\
        .format(name, current_time)
    np.save(os.path.join(out_dir, extractions_filename), extractions)
    print("Extractions shape: {}".format(extractions.shape))
    print("Load extractions with numpy: 'np.load('{}.npy')'"
          .format(extractions_filename))


def load_latest_results(out_dir):
    index_files = []
    extractions_files = []
    for root, directories, filenames in os.walk(out_dir):
        for filename in filenames:
            if filename.endswith(".{}".format('npy')):
                if filename.startswith('index_resnet50_'):
                    index_files.append(os.path.join(out_dir, filename))
                elif filename.startswith('extractions_resnet50_'):
                    extractions_files.append(os.path.join(out_dir, filename))
    index_files.sort()
    extractions_files.sort()
    if len(index_files) == 0 or len(extractions_files) == 0:
        return None, None
    else:
        print("\nLoading existing results:\n{}\n{}\n"
              .format(index_files[-1], extractions_files[-1]))
        return np.load(os.path.join(out_dir, index_files[-1])), \
               np.load(os.path.join(out_dir, extractions_files[-1]))
