import numpy as np

# ПРОВЕРИЛ
abc = "0123456789ABCEHKMOPTXY"
mapping = {
    'А': 'A',
    'В': 'B',
    'С': 'C',
    'Е': 'E',
    'Н': 'H',
    'К': 'K',
    'М': 'M',
    'О': 'O',
    'Р': 'P',
    'Т': 'T',
    'Х': 'X',
    'У': 'Y',
}

# ПРОВЕРИЛ
def is_valid_str(s, abc=abc):
    for ch in s:
        if ch not in abc:
            return False
    return True

# ПРОВЕРИЛ
def convert_to_eng(text, mapping=mapping):
    return ''.join([mapping.get(a, a) for a in text]) # изящная реализация - если элемент не найден, то берем его


# ПРОВЕРИЛ
def decode_sequence(pred, abc):
    """
    формат pred: 20 x B x alphabet_size. 
    Обрати внимание: тензор содержит alphabet_size! 
    - т.е. не единственный вариант, а все возможные варианты
    - это не распределение, а значения до soft_max или других вероятностных трансформаций
    """
    pred = pred.permute(1, 0, 2).cpu().data.numpy() #B x 20 x alphabet_size . 
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc)) # для каждого элемента батча вызывается pred_to_string
    return outputs # [ГРЗ_1, ГРЗ_2,.. ГРЗ_B]


# ПРОВЕРИЛ
def pred_to_string(pred, abc): # pred 20 x alphabet_size (см. комментарий в decode_sequence относительно alphabet_size)
    seq = []
    for i in range(len(pred)): # двигаемся побуквенно
        label = np.argmax(pred[i]) # выбираем букву с максимальным значением (это аналог вероятности, но не вероятность, т.к. soft max не применялся)
        seq.append(label - 1) # label - 1, потому что индексы смещены на 1 вперед (нулевой индекс отдан для разделителя)
    out = []
    for i in range(len(seq)):
        if len(out) == 0: # условие для первой итерации
            if seq[i] != -1: # если первый символ != разделитель
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]: # если символ != разделитель и символ != предшествующий символ
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out


# def labels_to_text(labels, abc=abc):
#     return ''.join(list(map(lambda x: abc[int(x) - 1], labels)))
#
#
# def text_to_labels(text, abc=abc):
#     return list(map(lambda x: abc.index(x) + 1, text))
