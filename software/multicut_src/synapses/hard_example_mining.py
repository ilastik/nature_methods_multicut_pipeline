import numpy as np
from sklearn.ensemble import RandomForestClassifier


# choose random subset of size n from the data
# returns the data points and chosen indices
def choose_random_subset(data, n):
    # restrict number of chosen points if n is bigger than the actual number of points
    n_ = min(n, len(data))
    choice = np.random.permutation(len(data))
    return data[choice[:n_]], choice[:n_]


# mine for hard training examples
# from the given negative and positive examples
# in the spirit of Felzenszwalb, adjusted from 'ilastik/op_defect_missing_data':
# https://github.com/soledis92/op_detect_missing_data
# NOTE in contrast to the ilastik implementation we do not parallelize over mining the
# positive and negative examples, because the random forest is already parallel
def mine_hard_examples_felzenszwalb(
    negative_examples,
    positive_examples,
    first_samples=250,
    max_remove_per_step=10,
    max_add_per_step=250,
    max_samples=5000,
    n_training_steps=4,
    classifier_opts={}
):

    # set up the random forest that will be trained
    rf = RandomForestClassifier(**classifier_opts)

    # draw the initial positive and negative samples
    init_negative, choice_negative = choose_random_subset(negative_examples, first_samples)
    init_positive, choice_positive = choose_random_subset(positive_examples, first_samples)

    # set up the lists of current negatives, positives
    # as well as indices draw so far and finished indicators
    samples = [negative_examples, positive_examples]
    current_examples = [init_negative, init_positive]
    choices = [choice_negative, choice_positive]
    finished = [False, False]

    # closure to perform single step of felseszwalb training
    def mining_step(data, chosen, index):

        # predict the current data and find the hard and easy examples
        current_prediction = rf.predict(data)

        hard_examples = np.where(current_prediction != index)[0]
        easy_examples = np.where(current_prediction == index)[0]

        # check if any easy examples were already included in the
        # previous chosen examples
        easy_already_chosen = np.setdiff1d(easy_examples, chosen) if len(easy_examples) > 0 else []

        # if any of the easy examples were already chosen in the
        # previous iteration (i.e. were hard examples), we remove up to 'max_remove_per_step' of them
        if len(easy_already_chosen) > 0:
            remove_from_chosen, _ = choose_random_subset(easy_already_chosen, max_remove_per_step)
            chosen = np.setdiff1d(chosen, remove_from_chosen)

        # next, we grow the chosen hard examples by adding 'max_add_per_step' of them
        n_hard_prev = len(chosen)
        add_to_chosen, _ = choose_random_subset(hard_examples, max_add_per_step)
        chosen = np.union1d(chosen, add_to_chosen)
        n_hard_added = len(chosen) - n_hard_prev

        # reduce the number of chosen samples if we exceed the max number of samples
        if len(chosen) > max_samples:
            chosen = choose_random_subset(chosen, max_samples)[0]

        return data[chosen], chosen, n_hard_added == 0

    # perform the mining for specified number of steps
    for k in range(n_training_steps):

        # fit the random forest to the current data
        current_labels = np.zeros(
            len(current_examples[0]) + len(current_examples[1]), dtype='uint8'
        )
        current_labels[len(current_examples[0]):] = 1
        rf.fit(
            np.concatenate(current_examples),
            current_labels
        )

        assert len(current_examples) == 2
        for i in range(2):
            new_examples, new_choice, new_finished = mining_step(samples[i], choices[i], i)
            current_examples[i] = new_examples
            choices[i] = new_choice
            finished[i] = new_finished

        # stop training if we have already found all hard examples
        if np.all(finished):
            break

    # return the indices of hard positive and negative examples
    return choices
