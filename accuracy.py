def token_lvl_accuracy(gt, pred):
    """
    gt = ground truth sequence
    pred = predicted sequence
    """
    correct = 0
    ground_t = gt.split(" ")
    prediction = pred.split(" ")

    longer = ground_t if len(ground_t) > len(prediction) else prediction
    shorter = prediction if len(ground_t) > len(prediction) else ground_t

    difference = len(longer) - len(shorter)
    shorter.extend(["" for _ in range(difference)])

    correct = sum([1 for i in range(len(longer)) if longer[i] == shorter[i]])
    # print(longer)
    # print(shorter)
    # print(correct)
    return int(correct) / len(shorter)  # same length as longer


def seq_lvl_accuracy(gt, pred):
    if len(gt) != len(pred):
        return 0

    if token_lvl_accuracy(gt, pred) == 1.0:
        return 1

    return 0
