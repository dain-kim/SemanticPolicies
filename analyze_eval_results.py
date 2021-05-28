from scipy.spatial import distance
import json
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

EVAL_RESULT_PATH = './val_result.json'

with open(EVAL_RESULT_PATH, 'r') as f:
    results = json.load(f)


# Object Detection Module
# for confusion matrix
all_objects_gt = []
all_objects_pred = []
# for hamming distance
all_hamming = []

# Attention Network Module
# for confusion matrix
all_attn_gt = []
all_attn_pred = []
# for hamming distance
all_attn_hamming = []

# Control Module
pick_total = 0
pick_success = 0
place_total = 0
place_success = 0


for eval_type in ['sorting', 'kitting']:
    print('======================================================\nTask Type:', eval_type)
    sample_results = list(results[eval_type].keys())
    for sample_name in sample_results:
        detection_results = results[eval_type][sample_name]['object_detection']
        attention_results = results[eval_type][sample_name]['attention']
        control_results = results[eval_type][sample_name]['control']

        ## Module: Object Detection
        '''
        detection = {
            'ground_truth': [0,0,0,17,17,21],         -- one red cup, two red bins
            'detected_objects': [0,0,0,0,17,21]       -- one red cup, one red bin detected
        }
        '''
        gt = detection_results['ground_truth']
        pred = detection_results['detected_objects']
        all_objects_gt.append(gt)
        all_objects_pred.append(pred)
        all_hamming.append(distance.hamming(gt, pred))
        # TODO

        ## Module: Attention Network
        '''
        attention = [
                {
                'keyword': 'red cup',                 -- command involving "red cup"
                'ground_truth': [0,0,0,0,21,21],      -- two red cups
                'attended_objects': [0,0,0,0,0,21]    -- one red cup attended to
            }
        ]
        '''
        for subtask_attn_results in attention_results:
            gt = subtask_attn_results['ground_truth']
            pred = subtask_attn_results['attended_objects']
            all_attn_gt.append(gt)
            all_attn_pred.append(pred)
            all_attn_hamming.append(distance.hamming(gt, pred))
        # TODO

        # Module: Control
        pick_total += control_results['pick_total']
        pick_success += control_results['pick_success']
        place_total += control_results['place_total']
        place_success += control_results['place_success']
        # TODO

    print('#### DETECTION ####')
    print('Avg Hamming Distance for', eval_type, np.mean(all_hamming))
    # print('Confusion Matrix for', eval_type)
    # print(metrics.classification_report(all_objects_gt, all_objects_pred))

    print('#### ATTENTION ####')
    print('Avg Hamming Distance for', eval_type, np.mean(all_attn_hamming))
    # print('Confusion Matrix for', eval_type)
    # print(metrics.classification_report(all_attn_gt, all_attn_pred))

    print('##### CONTROL #####')
    print("Pick Task Success Rate: {}/{} ({:.1f}%)".format(pick_success, pick_total, 100.0*pick_success/pick_total))
    print("Place Task Success Rate: {}/{} ({:.1f}%)".format(place_success, place_total, 100.0*place_success/place_total))



