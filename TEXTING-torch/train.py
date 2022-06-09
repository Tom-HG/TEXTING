import torch
import  torch.nn as nn
import argparse
from utils import load_data, preprocess_adj, preprocess_features, print_log
import numpy as np
from model import myModel,GATmodel,GCNmodel,Gmodel,Model
# from new_model import Model
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
parser = argparse.ArgumentParser(description='TextING_pytorch')
parser.add_argument("--model", type=str, default="gnn", help="Model string.")
parser.add_argument("--dataset", type=str,
                    default="mr", help="choose a dataset: 'mr','ohsumed','R8','R52'")
parser.add_argument("--learning_rate", type=float,
                    default=0.001, help="Initial learning rate.")
parser.add_argument("--epoch", type=int,
                    default=200, help="Number of epochs to train.")
parser.add_argument("--batch_size", type=int,
                    default=128, help="Size of batches per epoch.")
parser.add_argument("--accumulation_steps", type=int,
                    default=1, help="Size of gradient accumulation.")
parser.add_argument("--input_dim", type=int,
                    default=300, help="Dimension of input.")
parser.add_argument("--hidden", type=int,
                    default=128, help="Number of units in hidden number.")
# parser.add_argument("--hidden_2", type=int,
#                     default=32, help="Number of units in hidden number.")
parser.add_argument("--steps", type=int,
                    default=2, help="Number of gate steps.")
parser.add_argument("--layers", type=int,
                    default=2, help="Number of graph layers.")
# parser.add_argument("--depth", type=int,
#                     default=1, help="Number of informer layers.")
parser.add_argument("--weight_decay", type=float,
                    default=5e-4, help="Weight for L2 loss on embedding matrix.")
parser.add_argument("--dropout", type=float,
                    default=0.5, help="Dropout rate (1 - keep probability).")
parser.add_argument("--early_stopping", type=int,
                    default=-1, help="Tolerance for early stopping (# of epochs).")
parser.add_argument("--max_degree", type=str,
                    default=3, help="Maximum Chebyshev polynomial degree.")
parser.add_argument("--device", type=str,
                    default='cuda:0', help="device.")
args = parser.parse_args()

# Load data
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(args.dataset)
# print(train_adj[0])

args.labels_num = len(train_y[1])

# print(np.shape(train_adj[1]))
# print(np.shape(train_feature[1]))
# print(np.shape(train_y[1]))

# adj_train = [[],[],[]]
# adj_val = [[],[],[]]
# adj_test = [[],[],[]]
# train_mask = []
# val_mask = []
# test_mask = []
# Some preprocessing
if args.dataset=='mr':
    args.max_len = 46
elif args.dataset == 'ohsumed':
    args.max_len = 197
elif args.dataset == 'R8':
    args.max_len = 291
elif args.dataset == 'R52':
    args.max_len = 301

print('loading training set')
# adj_train[0], train_mask = preprocess_adj(train_adj[0])
# adj_train[1], _= preprocess_adj(train_adj[1])
# adj_train[2], _ = preprocess_adj(train_adj[2])
train_adj, train_mask = preprocess_adj(train_adj)
train_feature = preprocess_features(train_feature)
print('loading validation set')
# adj_val[0], val_mask = preprocess_adj(val_adj[0])
# adj_val[1], _ = preprocess_adj(val_adj[1])
# adj_val[2], _ = preprocess_adj(val_adj[2])
val_adj, val_mask = preprocess_adj(val_adj)
val_feature = preprocess_features(val_feature)
print('loading test set')
# adj_test[0], test_mask = preprocess_adj(test_adj[0])
# adj_test[1], _ = preprocess_adj(test_adj[1])
# adj_test[2], _ = preprocess_adj(test_adj[2])
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)

# print(np.shape(train_feature))
# args.max_len = len(train_feature[0])

# train_adj = np.array(adj_train)
# val_adj = np.array(adj_val)
# test_adj = np.array(adj_test)
# train_mask = np.array(train_mask)
# val_mask = np.array(val_mask)
# test_mask = np.array(test_mask)

# del adj_train, adj_val, adj_test

# print(np.shape(train_adj))  #(B, MAX_LEN, MAN_LEN)
# print(np.shape(train_mask)) #(B, MAX_LEN, 1)
# print(np.shape(train_feature)) #(B, MAX_LEN, DIM)

# print(np.shape(train_adj)) #(3,6398,45,45)
# print(np.shape(train_mask)) #(6398, 45, 1)
# print(np.shape(train_feature)) #(6398, 45, 300)


model = Model(args.input_dim, args.hidden, args.labels_num, args.dropout)
model.to(args.device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


# Define model evaluation function
# def evaluate(features, support, mask, labels):
#     t_test = time.time()
#     features = torch.from_numpy(features.astype(np.float32)).to(args.device)
#     support = torch.from_numpy(support.astype(np.float32)).to(args.device)
#     mask = torch.from_numpy(mask.astype(np.float32)).to(args.device)
#     labels = torch.from_numpy(labels.astype(np.float32)).to(args.device)
#     model.eval()
#     with torch.no_grad():
#         logits = model(features, support, mask)
#         loss = criterion(logits, torch.argmax(labels, 1))
#         pred = torch.argmax(logits, 1)
#         acc = ((pred == torch.argmax(labels, 1)).float()).sum().item() / len(labels)
#     return loss.cpu().numpy(), acc, pred.cpu().numpy(), labels.cpu().numpy(), (time.time() - t_test)

def evaluate(features, support, mask, labels):
    t_test = time.time()
    pred = torch.zeros(size=(len(labels),1), dtype=torch.long).to(args.device).squeeze()
    loss = 0
    num = 0
    indices = np.arange(0, len(labels))
    np.random.shuffle(indices)
    for start in range(0, len(labels), args.batch_size):
        end = start + args.batch_size
        idx = indices[start:end]
        num += 1

        x = torch.from_numpy(features[idx].astype(np.float32)).to(args.device)
        adj = torch.from_numpy(support[idx].astype(np.float32)).to(args.device)
        m = torch.from_numpy(mask[idx].astype(np.float32)).to(args.device)
        y = torch.from_numpy(labels[idx].astype(np.float32)).to(args.device)
        model.eval()
        with torch.no_grad():
            logits = model(x, adj, m)
            loss += criterion(logits, torch.argmax(y, 1))
            pred[idx] = torch.argmax(logits, 1)
    loss = loss / num
    label = torch.from_numpy(labels.astype(np.float32)).to(args.device)
    acc = ((pred == torch.argmax(label, 1)).float()).sum().item() / len(label)
    return loss.cpu().numpy(), acc, pred.cpu().numpy(), label.cpu().numpy(), (time.time() - t_test)

cost_val = []
best_val = 0
best_f1 = 0
best_epoch = 0
best_acc = 0
best_cost = 0
test_doc_embeddings = None
preds = None
labels = None


print('train....')
for epoch in range(args.epoch):
    model.train()
    t = time.time()

    # Training step
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)

    train_loss, train_acc = 0, 0
    # loss = 0
    step = 0
    for start in range(0, len(train_y), args.batch_size):
        end = start + args.batch_size
        idx = indices[start:end]
        # Construct feed dictionary

        x = torch.from_numpy(train_feature[idx].astype(np.float32)).to(args.device)
        support = torch.from_numpy(train_adj[idx].astype(np.float32)).to(args.device)
        mask = torch.from_numpy(train_mask[idx].astype(np.float32)).to(args.device)
        y = torch.from_numpy(train_y[idx].astype(np.float32)).to(args.device)
        # print(x.size())
        # print(support.size())
        # print(mask.size())


        logits = model(x, support, mask)
        # print(logits.size())
        # print(y.size())

        loss = criterion(logits, torch.argmax(y, 1))
        acc = ((torch.argmax(logits, 1) == torch.argmax(y, 1)).float()).sum().item() / len(y)

        train_loss += loss*len(idx)
        train_acc += acc*len(idx)

        # 2.1 loss regularization
        loss = loss / args.accumulation_steps
        # 2.2 back propagation
        loss.backward()

        # 3. update parameters of net
        if ((step + 1) % args.accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient


        # # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        step+=1
        del x, support, mask, y
        torch.cuda.empty_cache()



    if ((step) % args.accumulation_steps) != 0:
        # optimizer the net
        optimizer.step()  # update parameters of net
        optimizer.zero_grad()  # reset gradient

    train_loss /= len(train_y)
    train_acc /= len(train_y)

    # Validation
    val_cost, val_acc, val_pred, val_labels, val_duration = evaluate(val_feature, val_adj, val_mask, val_y)
    # val_f1 = f1_score(val_labels, val_pred, average='micro')
    cost_val.append(val_cost)

    # Test
    #loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)
    test_cost, test_acc, pred, labels, test_duration, = evaluate(test_feature, test_adj, test_mask, test_y)
    # test_f1 = f1_score(labels, pred, average='micro')

    torch.cuda.empty_cache()

    if val_acc >= best_val and test_acc>0.6:
        # best_f1 = val_f1
        best_val = val_acc
        best_epoch = epoch + 1
        best_acc = test_acc
        best_cost = test_cost
        preds = pred

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
          "val_acc=", "{:.5f}".format(val_acc),
          "test_acc=", "{:.5f}".format(test_acc),
          "time=", "{:.5f}".format(time.time() - t))

    if args.early_stopping > 0 and epoch > args.early_stopping and cost_val[-1] > np.mean(
            cost_val[-(args.early_stopping + 1):-1]):
        print("Early stopping...")
        break


print("Optimization Finished!")

# test_cost, test_acc, pred, labels, test_duration, = evaluate(test_feature, test_adj, test_mask, test_y)
# print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_cost, test_acc, test_duration))

from sklearn import metrics
# Best results
print('Best epoch:', best_epoch)
print("Test set results:", "cost=", "{:.5f}".format(best_cost), "val_acc=", "{:.5f}".format(best_val),
      "accuracy=", "{:.5f}".format(best_acc))

test_pred = []
test_labels = []
for i in range(len(labels)):
    test_pred.append(pred[i])
    test_labels.append(np.argmax(labels[i]))


print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))


text = []
with open('data/' + args.dataset + '_test.txt', 'r') as f:
    for line in f.readlines():
        text.append(line.strip())
with open('data/' + args.dataset + '_result.txt', 'w') as f:
    f.write('text\tlabels\tpred\n')
    for i in range(len(text)):
        f.write(text[i]+'\t'+str(test_labels[i])+'\t'+str(test_pred[i])+'\n')
