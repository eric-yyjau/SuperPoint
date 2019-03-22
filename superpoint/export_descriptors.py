import numpy as np
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import experiment
from superpoint.settings import EXPER_PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config

    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load(str(checkpoint))
        test_set = dataset.get_test_set()

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        while True:
            try:
                data = next(test_set)
            except dataset.end_set:
                break
            data1 = {'image': data['image']}
            data2 = {'image': data['warped_image']}
            pred1 = net.predict(data1, keys=['prob_nms', 'descriptors'])
            pred2 = net.predict(data2, keys=['prob_nms', 'descriptors'])
            pred = {'prob': pred1['prob_nms'],
                    'warped_prob': pred2['prob_nms'],
                    'desc': pred1['descriptors'],
                    'warped_desc': pred2['descriptors'],
                    'homography': data['homography']}
            ##### sparse
            sparse = True

            def getSparseFeatures(heatmap, desc, verbose=False):
                from models.model_wrap import SuperPointFrontend_torch
                fe = SuperPointFrontend_torch(weights_path='', nms_dist=4, conf_thresh=0.01, nn_thresh=0.7, load=False)
                points = fe.getPtsFromHeatmap(heatmap)
                print("pts: ", points.shape)
                def getSparseDesc(desc, pts):
                    return desc[pts[1,:].astype(int), pts[0, :].astype(int)]
                desc = getSparseDesc(desc, points)
                return points, desc
            if sparse:
                prob, desc = getSparseFeatures(pred['prob'], pred['desc'])
                """
                prob (3, N1)
                desc (N1, 256)
                """
                pred['prob'], pred['desc'] = prob.transpose(), desc
                prob, desc = getSparseFeatures(pred['warped_prob'], pred['warped_desc'])
                """
                prob (3, N1)
                desc (N1, 256)
                """
                pred['warped_prob'], pred['warped_desc'] = prob.transpose(), desc
                
            
            if not ('name' in data):
                pred.update(data)
            filename = data['name'].decode('utf-8') if 'name' in data else str(i)
            #### update pred
            pred.update({'image': data['image'].squeeze()})
            pred.update({'warped_image': data['warped_image'].squeeze()})
            
            filepath = Path(output_dir, '{}.npz'.format(filename))
            np.savez_compressed(filepath, **pred)
            i += 1
            pbar.update(1)
            if i == config['eval_iter']:
                break
