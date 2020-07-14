import argparse
import json
import os
import configparser
import pdb
from collections import defaultdict

import numpy as np
import torch

from person import PersonManager
from tracker import Tracker
from tracklet_manager import TrackletManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(BASE_DIR, "main.cfg"))

MIN_IOU = float(CONFIG['general']['iou'])
COS_DIST = float(CONFIG['general']['cos_distance'])
TRACKLET_LENGTH = int(CONFIG['general']['tracklet_length'])
NODE_COUNT = int(CONFIG['general']['node_count'])


def main(opt):
    track_dict = {}

    cam_range = opt.cams
    cam_range[1] += 1

    managers = {}

    for i in range(*cam_range):
        managers[i] = {}
        managers[i]['tracklet_manager'] = TrackletManager()

        open(os.path.join(BASE_DIR, f'results_{i}.txt'), "w").close()

        track_dict[i] = defaultdict(list)

    open(os.path.join(BASE_DIR, f'results_all.txt'), "w").close()

    model = torch.load(CONFIG['path']['model'])

    person_manager = PersonManager(model, node_count=NODE_COUNT)

    i = opt.start
    last_record_id = i
    while i <= opt.end:
        all_returns = []

        for c in range(*cam_range):
            features_and_detections = {}
            features = {}
            detections = {}
            try:
                with open(
                    os.path.join(CONFIG['path']['detections'], 
                    "camera{}/{}.json".format(c,i)
                    )
                ) as f:
                    detections = json.load(f)
            except FileNotFoundError:
                pass
                # print("INFO: No detections for camera", c, "frame", i)
                # raise FileNotFoundError
            
            try:
                with open(
                    os.path.join(
                        CONFIG['path']['features'], 
                        "camera{}/{}.json".format(c,i)
                    )
                ) as f:
                    features = json.load(f)
            except FileNotFoundError:
                pass
                # print("INFO: No features for camera", c, "frame ", i)
                # raise FileNotFoundError


            for f in features:
                if int(f["ID"]) > 0:
                    features_and_detections[int(f["ID"])] = {
                        "features": np.array(f["features"], dtype=np.float32)
                    }

            for d in detections:
                if int(d["ID"]) > 0:
                    features_and_detections[int(d["ID"])]["bbox"] = np.array([d["xmin"], d["ymin"], d["xmax"], d["ymax"]], 
                                                                    dtype=np.int32)

            # print(detections)
            # print(features)
            # print(features_and_detections)
            returns = managers[c]['tracklet_manager'].update(
                i,
                list(features_and_detections.values())
            )
            returns = list(filter(lambda x: x.get_final_features is not None, returns))
            all_returns.extend(returns)

            
            # print(returns)

            # if returns and (not returns[0].last_features is None):
                # print(person_manager.compute_score(None, returns[0]))
        
        if all_returns:
            person_matches = person_manager.update(all_returns)

            for ti, tracklet in enumerate(all_returns):
                # print(tracklet.tracklet_id)
                # print(len(tracklet.frame_ids), len(tracklet.x_values))
                # print(tracklet.frame_ids, tracklet.x_values)
                for fi, frame_id in enumerate(tracklet.frame_ids):
                    track_dict[tracklet.cam_id][frame_id].append(
                        (
                            str(tracklet.cam_id),
                            str(person_matches[ti].ID),
                            str(frame_id),
                            str(tracklet.x_values[fi]),
                            str(tracklet.y_values[fi]),
                            str(tracklet.w_values[fi]),
                            str(tracklet.h_values[fi])
                        )
                    )

            # cam_ids = list(map(lambda x: x.frame_ids, all_returns))
            # result_ids = list(map(lambda x: x.ID, person_matches))
            # print(track_dict)
            print(f'Frame {i}, Total Persons: {len(person_matches)}')

        if i % 600 == 0:
            for ci in range(*cam_range):
                with open(os.path.join(BASE_DIR, f'results_{ci}.txt'), "a") as fp:
                    
                    for lri in range(last_record_id, i-TRACKLET_LENGTH):
                        for rec in track_dict[ci][lri]:
                            string = ', '.join(rec)
                            string += ', -1, -1\n'
                            fp.write(string)

                        track_dict[ci].pop(lri)

            last_record_id = i - TRACKLET_LENGTH - 1

            print('Written until', last_record_id)

        i += 1

    for ci in range(*cam_range):
        with open(os.path.join(BASE_DIR, f'results_{ci}.txt'), "a") as fp:
            for lri in range(last_record_id, i+1):
                for rec in track_dict[ci][lri]:
                    string = ', '.join(rec)
                    string += ', -1, -1\n'
                    fp.write(string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BCS MTMCT')

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--trainingdataset", type=str, default='tl120_iou0.85_d0.3_processed_training.pickle')
    parser.add_argument("--valdataset", type=str, default='tl120_iou0.85_d0.3_processed_val.pickle')
    parser.add_argument("--output", type=str, default='dataset.pickle')
    parser.add_argument("--datafile", type=str, default='.')
    parser.add_argument("--cams", type=int, nargs='+', default=[1, 8])

    parser.add_argument("--start", type=int, default=45000)
    parser.add_argument("--end", type=int, default=390000)

    opt = parser.parse_args()

    main(opt)
