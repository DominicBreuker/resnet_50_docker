import os
import numpy as np
from datetime import datetime


def save_results(args, index, extractions, out_dir):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    index = np.array(index)
    extractions = np.array(extractions)

    index_filename = "index_resnet50_{}x{}_{}_{}"\
        .format(args.height, args.width, args.mode, current_time)
    np.save(os.path.join(out_dir, index_filename), index)
    print("Load index with numpy: 'np.load('{}.npy')'"
          .format(index_filename))

    extractions_filename = "extractions_resnet50_{}x{}_{}_{}"\
        .format(args.height, args.width, args.mode, current_time)
    np.save(os.path.join(out_dir, extractions_filename), extractions)
    print("Load extractions with numpy: 'np.load('{}.npy')'"
          .format(extractions_filename))
