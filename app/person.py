from collections import namedtuple
import pdb
import math

import dgl
import networkx as nx
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


BCSBatch = namedtuple('BCSBatch', ['graph'])
MockTracklet = namedtuple('MockTracklet', ['last_features'])

SCORE_THRESHOLD = 0.4
SCORE_DEFAULT = 100000
FORGET_FRAME = 6000


MARGIN_INTRA = 600
MARGIN_INTER = 6000


class Person():
    def __init__(self, ID, tracklet, update_id, node_count=4):
        self.ID = ID
        self.node_count = node_count
        self.max_edge_count = self.node_count*self.node_count
        self.tracklet_count = 0
        
        # TODO: create graph
        self.graph = nx.DiGraph()
        
        for i in range(node_count):
            self.graph.add_node(i)

        # for i in range(node_count):
        #     for j in range(node_count):
        #         self.graph.add_edge(i, j)
        #         self.graph.add_edge(j, i)

        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        self.graph = dgl.DGLGraph(self.graph)
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())
        self.batched_graph = BCSBatch(graph=dgl.batch([self.graph]))
        # pdb.set_trace()
        self.node_features = []
        self.node_tracklets = []
        self.unique_tracklets = []

        self.start_id = update_id
        self.update_id = update_id

        self.put_feature_vector(tracklet)

        # self.add_tracklet(tracklet_feature)

    def put_feature_vector(self, tracklet):

        if len(self.unique_tracklets) == self.node_count:
            self.unique_tracklets.pop(0)

        self.unique_tracklets.append(tracklet)

        # self.node_features.clear()
        self.node_tracklets.clear()

        self.tracklet_count = min(self.tracklet_count + 1, self.node_count)
    
        tracklet_occurence = math.floor(self.node_count/self.tracklet_count)

        for i in range(len(self.unique_tracklets)):
            for _ in range(tracklet_occurence):
                self.node_tracklets.append(self.unique_tracklets[i])

        for _ in range(self.node_count - tracklet_occurence*len(self.unique_tracklets)):
            self.node_tracklets.append(tracklet)            

        self.node_features = list(map(
            lambda x: x.last_features, 
            self.node_tracklets
        ))

        self.graph.remove_edges(list(range(len(self.graph.edges()[0]))))

        for i in range(len(self.node_tracklets)):
            for j in range(i+1, len(self.node_tracklets)):
                if self.node_tracklets[i].cam_id == self.node_tracklets[j].cam_id:
                    if abs(self.node_tracklets[i].last_frame_id - self.node_tracklets[j].last_frame_id) < MARGIN_INTRA:
                        self.graph.add_edge(i, j)
                        self.graph.add_edge(j, i)
                else:
                    if abs(self.node_tracklets[i].last_frame_id - self.node_tracklets[j].last_frame_id) < MARGIN_INTER:
                        self.graph.add_edge(i, j)
                        self.graph.add_edge(j, i)
        # print(len(self.graph.edges()[0]))

        # # TODO: HOLD OLD NODES FOR LONGER

        # if self.tracklet_count == 0:
        #     for i in range(self.node_count):
        #         self.node_features.append(feature_vector)

        # elif self.tracklet_count == 1:
        #     for i in range(2):
        #         self.node_features.pop(0)
        #         self.node_features.append(feature_vector)

        # elif self.tracklet_count == 2:
        #     for i in range(2):
        #         self.node_features.pop(1)
        #         self.node_features.append(feature_vector)

        # elif self.tracklet_count == 3:
        #     self.node_features.pop()
        #     self.node_features.append(feature_vector)

        # else:
        #     self.node_features.pop(0)
        #     self.node_features.append(feature_vector)            


    def update(self, update_id, tracklet):
        self.update_id = update_id

        self.put_feature_vector(tracklet)


class PersonManager():
    def __init__(self, model, node_count=4):
        self.node_count = node_count
        self.persons = []
        self.model = model
        self.next_person_id = 1

        self.model.eval()

        self.update_id = 1

        # self.populate_mock()

    def populate_mock(self):
        for i in range(5):
            rv = np.random.rand(1, 128)
            # print(rv)
            self.create_person(
                MockTracklet(
                    last_features = rv
                )
            )

    def create_person(self, tracklet):
        new_person = Person(
            self.next_person_id,
            tracklet,
            self.update_id,
            self.node_count
        )
        self.persons.append(new_person)

        self.next_person_id += 1

        return new_person

    def update_person(self, person, tracklet):
        return person.update(self.update_id, tracklet)

    def compute_score(self, person, tracklet):

        node_features = torch.FloatTensor([person.node_features]).cuda() if torch.cuda.is_available() else torch.FloatTensor([person.node_features])
        tracklet_features = torch.FloatTensor([tracklet.last_features]).cuda() if torch.cuda.is_available() else torch.FloatTensor([person.node_features])
        
        with torch.no_grad():
            logits = self.model(
                person.batched_graph, 
                node_features, 
                tracklet_features
            )

        return float(logits[0])

    def update(self, tracklets):
        person_matches = [None]*len(tracklets)

        if not self.persons:
            person_matches = list(map(lambda t: self.create_person(t), tracklets))
        else:
            score_matrix = SCORE_DEFAULT*np.ones((len(tracklets), len(self.persons)), dtype=np.float32)

            no_match_tracklet_ids = []
            match_tracklet_ids = list(range(len(tracklets)))
            for i, tracklet in enumerate(tracklets):
                for j, person in enumerate(self.persons):
                    tp_score = self.compute_score(person, tracklet)
                    # print(tp_score)
                    if tp_score > SCORE_THRESHOLD:
                        score_matrix[i][j] = 1/tp_score

                if np.sum(score_matrix[i, :]) >= SCORE_DEFAULT*len(self.persons):
                    no_match_tracklet_ids.append(i)
            
            no_match_tracklet_ids.sort(reverse=True)

            for i in range(len(no_match_tracklet_ids)):
                score_matrix = np.delete(
                    score_matrix,
                    no_match_tracklet_ids[i],
                    0
                )
                match_tracklet_ids.pop(no_match_tracklet_ids[i])

            real_match_tracklet_indices = []
            if match_tracklet_ids:
                tracklet_i_indices, person_indices = linear_sum_assignment(score_matrix)
                for i in range(len(tracklet_i_indices)):
                    self.update_person(
                        self.persons[int(person_indices[i])], 
                        tracklets[match_tracklet_ids[tracklet_i_indices[i]]]
                    )
                    real_match_tracklet_indices.append(tracklet_i_indices[i])
                    person_matches[match_tracklet_ids[tracklet_i_indices[i]]] = self.persons[int(person_indices[i])]
                # print(i, score_matrix[i, :])
            real_match_tracklet_indices.sort(reverse=True)

            for i in range(len(real_match_tracklet_indices)):
                match_tracklet_ids.pop(real_match_tracklet_indices[i])

            no_match_tracklet_ids.extend(match_tracklet_ids)

            for i in range(len(no_match_tracklet_ids)):
                created_person = self.create_person(tracklets[no_match_tracklet_ids[i]])
                    
                person_matches[no_match_tracklet_ids[i]] = created_person

            no_match_tracklet_ids.sort(reverse=True)

            remove_ids = []
            for i, p in enumerate(self.persons):
                if (self.update_id - p.update_id) > FORGET_FRAME:
                    remove_ids.append(i)

            remove_ids.sort(reverse=True)

            for i in range(len(remove_ids)):
                self.persons.pop(remove_ids[i])

            # TODO: return tracklet ids
            self.update_id += 1

        return person_matches