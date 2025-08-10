def decode_predictions(preds, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    pred_texts = []
    for pred in preds:
        string = ''
        for i in range(len(pred)):
            if pred[i] != -1 and (i == 0 or pred[i] != pred[i - 1]):
                string += alphabet[pred[i]]
        pred_texts.append(string)
    return pred_texts
