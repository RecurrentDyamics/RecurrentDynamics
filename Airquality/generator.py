import ujson as json
import numpy as np
import pandas as pd

seq_len = 36

config = json.load(open('./config'))

ground = pd.read_csv('./csv/ground.csv', parse_dates = ['datetime'])
missing = pd.read_csv('./csv/missing.csv', parse_dates = ['datetime'])

cols = ['0010%02d' % i for i in range(1, 36 + 1)]

start = 0
stride = 3

def parse_val(val):
    val = (val[cols] - config['mean']) / config['std']
    val = val.fillna(0.0)
    return val

def parse_mask(val):
    return ~pd.isnull(val[cols])

def train_gen(ground, missing, direction = 'forward'):
    if direction == 'forward':
        ground = ground.sort_values(by = ['datetime'])
        missing = missing.sort_values(by = ['datetime'])
    else:
        ground = ground.sort_values(by = ['datetime'], ascending = False)
        missing = missing.sort_values(by = ['datetime'], ascending = False)

    values = []
    masks = []
    deltas = []

    labels = []
    label_masks = []

    years = []
    months = []
    days = []
    hours = []

    eval_flag = False

    for i in range(ground.shape[0]):
        gv = parse_val(ground.iloc[i])
        mv = parse_val(missing.iloc[i])

        gb = parse_mask(ground.iloc[i])
        mb = parse_mask(missing.iloc[i])

        # We use mb (missing bool) and mv (missing value) as the input, but we also record the ground truth to be predicted
        # gb is set to be true, if ground mask is not equal to the missing mask
        gb = gb ^ mb

        values.append(mv.tolist())

        if i == 0:
            delta = np.zeros(36)
        else:
            delta = np.ones(36) + (1 - np.array(masks[-1])) * np.array(deltas[-1])

        deltas.append(delta.tolist())
        masks.append(mb.map(lambda x: 1 if x else 0).tolist())

        labels.append(gv.tolist())
        label_masks.append(gb.map(lambda x: 1 if x else 0).tolist())

        year = ground.iloc[i]['datetime'].year
        month = ground.iloc[i]['datetime'].month
        day = ground.iloc[i]['datetime'].day
        hour = ground.iloc[i]['datetime'].hour

        years.append(year)
        months.append(month)
        days.append(day)
        hours.append(hour)

        if gb.sum() > 0 and month in [3, 6, 9, 12]:
            eval_flag = True

    return {'years': years, 'months': months, 'days': days, 'hours': hours, \
            'values': values, 'masks': masks, 'deltas': deltas, \
            'labels': labels, 'label_masks': label_masks}, eval_flag

train_fs = open('./data/train', 'w')
eval_fs = open('./data/eval', 'w')

while 1:
    if start + seq_len - 1 >= ground.shape[0]: break
    content_f, eval_flag = train_gen(ground[start: start + seq_len], missing[start: start + seq_len], direction = 'forward')
    content_b, eval_flag = train_gen(ground[start: start + seq_len], missing[start: start + seq_len], direction = 'backward')

    content = json.dumps({'forward': content_f, 'backward': content_b})

    train_fs.write('%s\n' % content)
    if eval_flag:
        eval_fs.write('%s\n' % content)

    start += stride
