import logging
import timeit
import json
import numpy as np
import torch
import torch.optim as optim

from utilities.constants import CONV_E, TRANS_E, TRANS_H, TRANS_D, TRANS_R

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def train_model(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed):
    model_name = kg_embedding_model.model_name

    if model_name in [TRANS_E, TRANS_H, TRANS_D, TRANS_R]:
        return train_trans_x_model(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed)

    if model_name == CONV_E:
        return train_conv_e_model(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed)


def train_trans_x_model(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed):
    np.random.seed(seed=seed)
    indices = np.arange(pos_triples.shape[0])
    np.random.shuffle(indices)
    pos_triples = pos_triples[indices]

    kg_embedding_model = kg_embedding_model.to(device)

    optimizer = optim.Adagrad(kg_embedding_model.parameters(), lr=learning_rate)

    total_loss = 0
    loss_per_epoch = []

    log.info('****Run Model On %s****' % str(device).upper())

    model_params = kg_embedding_model.parameters()

    subjects = pos_triples[:, 0:1]
    objects = pos_triples[:, 2:3]

    neg_trips = []

    for epoch in range(num_epochs):
        start = timeit.default_timer()
        pos_batches = split_list_in_batches(input_list=pos_triples, batch_size=batch_size)
        current_epoch_loss = 0.

        for i in range(len(pos_batches)):
            optimizer.zero_grad()
            pos_batch = pos_batches[i]
            current_batch_size = len(pos_batch)
            batch_subjs = pos_batch[:, 0:1]
            batch_preds = pos_batch[:, 1:2]
            batch_objs = pos_batch[:, 2:3]

            num_subj_corrupt = len(pos_batch) // 2
            num_obj_corrupt = len(pos_batch) - num_subj_corrupt
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

            corrupted_subj_indices = np.random.choice(np.arange(0, len(pos_batch)), size=num_subj_corrupt)
            corrupted_subjects = subjects[corrupted_subj_indices]

            subject_based_corrupted_triples = np.concatenate(
                [corrupted_subjects, batch_preds[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

            corrupted_obj_indices = np.random.choice(np.arange(0, len(pos_batch)), size=num_obj_corrupt)
            corrupted_objects = objects[corrupted_obj_indices]

            object_based_corrupted_triples = np.concatenate(
                [batch_subjs[num_subj_corrupt:], batch_preds[num_subj_corrupt:], corrupted_objects], axis=1)

            neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

            # for neg_elem in neg_batch:
            #     neg_trips.append(neg_elem.numpy().tolist())


            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            # model.zero_grad()
            # When to use model.zero_grad() and when optimizer.zero_grad() ?


            # kg_embedding_model.zero_grad()

            loss = kg_embedding_model(pos_batch, neg_batch)
            current_epoch_loss += (loss.item() * current_batch_size)

            # model_params = kg_embedding_model.parameters()
            #             # for p in model_params:
            #             #     print(p.grad)
            #             # print('################')

            loss.backward()
            optimizer.step()
            sum_grads = []
            sum_ws = []
            for p in kg_embedding_model.parameters():
                # print(p.shape)
                # print(torch.sum(p))
                if p.grad is not None:
                    sum_grads.append(torch.sum(torch.abs(p.grad)))
                sum_ws.append(torch.sum(torch.abs(p)))


            sum_grads = torch.tensor(sum_grads)
            sum_ws = torch.tensor(sum_ws)
            # print(torch.sum(sum_w))
            log.info("Absoulte sum of grads in epoch %d for batch %d is %f" % (epoch,i,np.sum(np.array(sum_grads))))
            log.info("Absolute sum of weights in epoch %d for batch %d is %f" % (epoch,i,np.sum(np.array(sum_ws))))
            log.info("Loss in epoch %d for batch %d is %f" % (epoch, i, loss.item()))

            # log.info("+++++++


            # for p in kg_embedding_model.parameters():
            #     if p.grad is not None:
            #         # print("Param: ", p)
            #         # print("Gradient: ", p.grad)
            #         # print("Shape of gradient:", p.grad.shape)
            #         # print(p.grad[0].numpy().tolist())
            #         if p.grad[0].numpy().tolist() != [0.,0.,0.,0.] or p.grad[1].numpy().tolist() != [0.,0.,0.,0.]:
            #             print("Good: ", p.grad[0].numpy().tolist())


            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        stop = timeit.default_timer()
        log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
        # Track epoch loss
        loss_per_epoch.append(current_epoch_loss/len(pos_triples))

    return kg_embedding_model, loss_per_epoch


def train_conv_e_model(kg_embedding_model, learning_rate, num_epochs, batch_size, pos_triples, device, seed):
    np.random.seed(seed=seed)
    indices = np.arange(pos_triples.shape[0])
    np.random.shuffle(indices)
    pos_triples = pos_triples[indices]

    # Create labels
    subject_relation_pairs = pos_triples[:, 0:2]
    entities = np.arange(kg_embedding_model.num_entities)
    labels = []

    for tuple in subject_relation_pairs:
        indices_duplicates = (subject_relation_pairs == tuple).all(axis=1).nonzero()
        objects = pos_triples[indices_duplicates, 2:3]
        objects = np.unique(np.ndarray.flatten(objects))
        label_vec = np.in1d(entities, objects) * 1
        labels.append(label_vec)

    kg_embedding_model = kg_embedding_model.to(device)
    optimizer = optim.SGD(kg_embedding_model.parameters(), lr=learning_rate)
    total_loss = 0
    loss_per_epoch = []

    log.info('****Run Model On %s****' % str(device).upper())
    # Train
    for epoch in range(num_epochs):
        start = timeit.default_timer()
        pos_batches = split_list_in_batches(input_list=subject_relation_pairs, batch_size=batch_size)
        label_batches = split_list_in_batches(input_list=labels, batch_size=batch_size)
        current_epoch_loss = 0.

        for i in range(len(pos_batches)):
            optimizer.zero_grad()
            pos_batch = pos_batches[i]
            label_batch = label_batches[i]
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)
            label_batch = torch.tensor(label_batch, dtype=torch.float, device=device)

            predictions = kg_embedding_model(pos_batch[:, 0:1], pos_batch[:, 1:2])
            loss = kg_embedding_model.compute_loss(pred=predictions, targets=label_batch)
            loss.backward()
            optimizer.step()
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
            current_epoch_loss += loss.item()
        stop = timeit.default_timer()
        log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
        loss_per_epoch.append(current_epoch_loss)

    return kg_embedding_model, loss_per_epoch

# if __name__ == '__main__':
#     triples = np.array([[1, 0, 2], [1, 0, 3], [2, 4, 4]])
#     subject_relations = np.array([[1, 0], [1, 0], [2, 4]])
#     hits = []
#     entities = np.array([0,1,2,3,4],dtype=np.int)
#
#
#     for r in subject_relations:
#         i = (subject_relations == r).all(axis=1).nonzero()
#         objects = triples[i, 2:3]
#         objects = np.unique(np.ndarray.flatten(objects))
#         print(objects)
#
#         t = np.in1d(entities,objects)*1
#         print(t)
#         print()
