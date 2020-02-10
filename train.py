import os
import pdb
import random
import pickle

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from model import BCS
from utils import EarlyStopping

torch.set_printoptions(threshold=5000)

MARGIN_INTRA = 600
MARGIN_INTER = 6000
GRAPH_SIZE = 4


# Used for creating datafile from tacklet records, for once.
def load_data(args):
    frame_width = 1920
    frame_height = 1080
    dataset = None

    with open(args.datafile, 'rb') as f:
        dataset = pickle.load(f)

    features = dataset['features']
    data = dataset['data']

    x = []
    graphs = []
    graph_data = []
    tracklet_features = []
    labels = []

    for k in data:
        if len(data[k]) > GRAPH_SIZE:
            data[k] = sorted(data[k], key = lambda x: x['start'])

            for i1 in range(len(data[k]) - GRAPH_SIZE):
                g = nx.DiGraph()
                graph_features = [None] * GRAPH_SIZE
                graph_features[0] = features[data[k][i1]['feature_id']]

                for i in range(GRAPH_SIZE):
                    g.add_node(i)

                for i2 in range(i1+1, i1+GRAPH_SIZE):
                    ii = i2 - i1
                    graph_features[ii] = features[data[k][i2]['feature_id']]
                    time_diff = data[k][i2]['start'] - data[k][i1]['end']
                    if data[k][i1]['cam'] != data[k][i2]['cam']:
                        if time_diff < MARGIN_INTER:
                            g.add_edge(0, ii)
                            g.add_edge(ii, 0)
                    else:
                        if time_diff < MARGIN_INTRA:
                            g.add_edge(0, ii)
                            g.add_edge(ii, 0)

                    for i3 in range(i2+1, i1+GRAPH_SIZE):
                        iii = i3 - i1
                        time_diff = data[k][i3]['start'] - data[k][i2]['end']
                        if data[k][i2]['cam'] != data[k][i3]['cam']:
                            if time_diff < MARGIN_INTER:
                                g.add_edge(iii, ii)
                                g.add_edge(ii, iii)
                        else:
                            if time_diff < MARGIN_INTRA:
                                g.add_edge(iii, ii)
                                g.add_edge(ii, iii)

                g.remove_edges_from(nx.selfloop_edges(g))
                g = DGLGraph(g)
                g.add_edges(g.nodes(), g.nodes())

                positive_features = features[random.choice(data[k][i1+GRAPH_SIZE:])['feature_id']]

                negative_features = None
                while True:
                    r = random.choice(list(data.keys()))

                    if r == k:
                        continue

                    negative_features = features[random.choice(data[r])['feature_id']]

                    break

                x.append((g, torch.FloatTensor(graph_features), torch.FloatTensor(positive_features)))
                graphs.append(g)
                graph_data.append(graph_features)
                tracklet_features.append(positive_features)
                labels.append(1)

                x.append((g, torch.FloatTensor(graph_features), torch.FloatTensor(negative_features)))
                graphs.append(g)
                graph_data.append(graph_features)
                tracklet_features.append(negative_features)
                labels.append(-1)


    set_ids = list(range(len(labels)))
    random.shuffle(set_ids)

    output_data = {
        'x': x,
        'labels': torch.LongTensor(labels),
        'set_ids': set_ids,
        'graphs': graphs,
        'graph_data': graph_data,
        'tracklet_features': tracklet_features
    }

    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)

    return output_data


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):

    with open("output.txt", "a") as fp:
        fp.write(str(args))
        fp.write('\n')

    # load_data(args)
    # return
    dataset = None

    with open(args.output, 'rb') as f:
        dataset = pickle.load(f)

    ####### DATASET KEYS
    # 'labels': 1/-1,
    # 'set_ids': shuffled ids,
    # 'graphs': graphs
    # 'graph_data': graph node features
    # 'tracklet_features': tracklet_features


    training_set_limit = int(len(dataset['set_ids'])*0.9)
    training_graph_data = []
    training_tracklet_data = []
    training_x = []
    training_labels = []
    test_graph_data = []
    test_tracklet_data = []
    test_labels = []

    for i in dataset['set_ids'][:training_set_limit]:
        training_graph_data.append(dataset['graph_data'][i])
        training_tracklet_data.append(dataset['tracklet_features'][i])
        training_x.append(dataset['x'][i])
        training_labels.append(dataset['labels'][i])

    for i in dataset['set_ids'][training_set_limit:]:
        test_graph_data.append(dataset['graph_data'][i])
        test_tracklet_data.append(dataset['tracklet_features'][i])
        # test_x.append(dataset['x'][i])
        test_labels.append(dataset['labels'][i])

    training_labels = torch.LongTensor(training_labels)
    test_labels = torch.LongTensor(test_labels)
    training_graph_data = torch.FloatTensor(training_graph_data)
    training_tracklet_data = torch.FloatTensor(training_tracklet_data)

    torch.cuda.empty_cache()
    # features = torch.FloatTensor(features)
    # labels = torch.LongTensor(labels)
    # if hasattr(torch, 'BoolTensor'):
    #     train_mask = torch.BoolTensor(train_mask)
    #     val_mask = torch.BoolTensor(val_mask)
    #     test_mask = torch.BoolTensor(test_mask)
    # else:
    #     train_mask = torch.ByteTensor(train_mask)
    #     val_mask = torch.ByteTensor(val_mask)
    #     test_mask = torch.ByteTensor(test_mask)
    # num_feats = features.shape[1]
    # n_classes = num_labels
    # n_edges = g.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d 
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #        train_mask.int().sum().item(),
    #        val_mask.int().sum().item(),
    #        test_labels.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        training_graph_data = training_graph_data.cuda()
        training_tracklet_data = training_tracklet_data.cuda()
        training_labels = training_labels.cuda()
        test_labels = test_labels.cuda()
    # create model
    
    model = BCS(g=training_x[0][0])
    model(training_graph_data[:2], training_tracklet_data[:2])


    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()

    loss_func = torch.nn.SoftMarginLoss()

    # loss_func = torch.nn.MultiMarginLoss()
    # loss_func = torch.nn.CrossEntropyLoss()

    # pdb.set_trace()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(training_x)

        loss = loss_func(logits, training_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        print(loss.item())
        # train_acc = accuracy(logits[train_mask], labels[train_mask])

        # if args.fastmode:
        #     val_acc = accuracy(logits[val_mask], labels[val_mask])
        # else:
        #     val_acc = evaluate(model, features, labels, val_mask)
        #     if args.early_stop:
        #         if stopper.step(val_acc, model):   
        #             break

        if epoch%15 == 0:
            torch.save(model, "model.pt")

        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
        #       " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
        #       format(epoch, np.mean(dur), loss.item(), train_acc,
        #              val_acc, n_edges / np.mean(dur) / 1000))

        # with open("output.txt", "a") as fp:
        #     fp.write("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
        #       " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}\n".
        #       format(epoch, np.mean(dur), loss.item(), train_acc,
        #              val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    

    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')

    parser.add_argument("--dataset", type=str, default='.')
    parser.add_argument("--output", type=str, default='dataset.pickle')
    parser.add_argument("--datafile", type=str, default='.')
    parser.add_argument("--cams", type=int, nargs='+', default=[1, 8])

    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)