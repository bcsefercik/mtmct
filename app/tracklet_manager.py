import statistics

import numpy as np
from scipy.optimize import linear_sum_assignment

WIDTH = 1920
HEIGHT = 1080
MAX_AGE = 600


class Tracklet:
    def __init__(self, tracklet_id, frame_id, bbox, features, cam_id):
        self.tracklet_id = tracklet_id
        self.cam_id = cam_id
        self.bboxes = [bbox]  # x,y,w,h
        self.last_bbox = bbox  # x1,y1,x2,y2
        self.x_values = [bbox[0]]
        self.y_values = [bbox[1]]
        self.w_values = [bbox[2] - bbox[0]]
        self.h_values = [bbox[3] - bbox[1]]
        self.last_features = features
        self.tracklet_features = features
        self.first_frame_id = frame_id
        self.last_frame_id = frame_id
        self.frame_ids = [frame_id]

    def update(self, frame_id, bbox, features):
        self.update_bbox(bbox)

        self.frame_ids.append(frame_id)

        # Empirical mean
        alpha = 1/len(self.frame_ids)
        self.last_features += alpha * (features - self.last_features)

        self.last_frame_id = frame_id


    def update_bbox(self, new_bbox):
        # print(new_bbox)
        self.last_bbox = new_bbox

        self.x_values.append(new_bbox[0])
        self.y_values.append(new_bbox[1])
        self.w_values.append(new_bbox[2] - new_bbox[0])
        self.h_values.append(new_bbox[3] - new_bbox[1])


    def get_bbox(self):
        result = [None] * 4

        result[0] = float(statistics.median(self.x_values)/WIDTH)
        result[1] = float(statistics.median(self.y_values)/HEIGHT)
        result[2] = float(statistics.median(self.w_values)/WIDTH)
        result[3] = float(statistics.median(self.h_values)/HEIGHT)

        return result


    def get_final_features(self, cam_id=None, bbox=False):
        final_features = []

        if cam_id is not None:
            final_features.extend(cam_id)

        if bbox:
            final_features.extend(self.get_bbox)

        final_features.extend(list(self.last_features))


    def length(self):
        return len(self.frame_ids)


class TrackletManager():
    def __init__(self, min_iou=0.7, max_cosine_distance=0.3, max_tracklet_length=10, cam_id=1):
        self.min_iou = min_iou
        self.max_cosine_distance = max_cosine_distance
        self.max_tracklet_length = max_tracklet_length
        self.cam_id = cam_id

        self.tracklets = []
        self.next_tracklet_id = 1
        self.frame_id = 1

    def init_tracklet(self, bbox, features):
        self.tracklets.append(Tracklet(
            self.next_tracklet_id,
            self.frame_id,
            bbox,
            features,
            self.cam_id
        ))

        self.next_tracklet_id += 1

    def is_tracklet_done(self, tracklet):
        return tracklet.length() >= self.max_tracklet_length or (self.frame_id-tracklet.last_frame_id) > MAX_AGE

    def update(self, frame_id, features_and_detections):
        self.frame_id = frame_id
        popped_tracklets = []

        if not self.tracklets:
            for di, dv in enumerate(features_and_detections):
                self.init_tracklet(
                    dv["bbox"],
                    dv["features"]
                )
        else:
            cost_matrix = []
            index_matches = []

            for di, dv in enumerate(features_and_detections):
                index_matches.append(di)
                cosine_distances = [1000.0] * len(self.tracklets)

                for i, tracklet in enumerate(self.tracklets):
                    if self.compute_iou(
                        dv["bbox"],
                        tracklet.last_bbox
                    ) > self.min_iou:
                        cosine_distances[i] = self.compute_cosine_dist(
                            dv["features"],
                            tracklet.last_features
                        )
                        if cosine_distances[i] > self.max_cosine_distance:
                            cosine_distances[i] += 1000

                cost_matrix.append(cosine_distances)

            if len(cost_matrix) > 0:
                cost_matrix = np.array(cost_matrix, dtype=np.float32)
                indices = linear_sum_assignment(cost_matrix)

                for ij in range(len(indices[0])):
                    detection_index = indices[0][ij]
                    tracklet_index = indices[1][ij]

                    if cost_matrix[detection_index][tracklet_index] < self.max_cosine_distance:
                        self.tracklets[tracklet_index].update(
                            self.frame_id,
                            features_and_detections[index_matches[detection_index]]["bbox"],
                            features_and_detections[index_matches[detection_index]]["features"]
                        )

                        index_matches[detection_index] = -1

            # Initialize new tracklets for no-match detections
            for ii, vv in enumerate(index_matches):
                if vv != -1:
                    self.init_tracklet(
                        features_and_detections[vv]["bbox"],
                        features_and_detections[vv]["features"]
                    )

            # Pop finished tracklets
            tracklets_to_pop = []
            for ti, tracklet in enumerate(self.tracklets):
                if self.is_tracklet_done(tracklet):
                    tracklets_to_pop.append(ti)

            tracklets_to_pop.sort(reverse=True)

            for pi in range(len(tracklets_to_pop)):
                popped_tracklets.append(
                    self.tracklets.pop(
                        tracklets_to_pop[pi]
                    )
                )

        return popped_tracklets

    def compute_iou(self, boxA, boxB):
        if isinstance(boxA, type(None)) or isinstance(boxB, type(None)):
            return 1.0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def area2d(self, b):
        return (b[2]-b[0])*(b[3]-b[1])

    def iou2d(self, b1, b2):
        ov = self.overlap2d(b1, b2)
        return ov / (self.area2d(b1) + self.area2d(b2) - ov)

    def compute_cosine_dist(self, emb1, emb2):
        sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
        return 1.0-float(sim)

