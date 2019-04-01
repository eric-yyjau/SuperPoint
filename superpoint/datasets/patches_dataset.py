import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.models.homographies import sample_homography
from superpoint.settings import DATA_PATH


class PatchesDataset(BaseDataset):
    default_config = {
        'dataset': 'hpatches',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': True
        }
    }

    def _init_dataset(self, **config):
        dataset_folder = 'COCO/patches' if config['dataset'] == 'coco' else 'HPatches'
        base_path = Path(DATA_PATH, dataset_folder)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        print("output alteration: ", config['alteration'])
        for path in folder_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            num_images = 1 if config['dataset'] == 'coco' else 5
            file_ext = '.ppm' if config['dataset'] == 'hpatches' else '.jpg'
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
            warped_image_paths = warped_image_paths[:config['truncate']]
            homographies = homographies[:config['truncate']]
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies}
        _ = None
        return files, _

    def _get_data(self, files, split_name, **config):
#         def _read_image(path):
#             sizer = np.array(config['preprocessing']['resize'])
# #             image = cv2.imread(path.decode('utf-8'))
#             image = cv2.imread(path.decode('utf-8'))
#             print("image pre-processing: ", image.shape)
#
#             s = max(sizer /image.shape[:2])
#             print("s: ", s)
#
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#             print("image convert gray: ", image.shape)
#             image = image[:int(sizer[0]/s),:int(sizer[1]/s)]
#             image = cv2.resize(image, (sizer[1], sizer[0]),
#                                      interpolation=cv2.INTER_AREA)
# #             image = image.astype('float32') / 255.0
#             print("image post-processing: ", image.shape)
#             return image[:,:,np.newaxis]
#
#         def _preprocess(image):
# #             tf.Tensor.set_shape(image, [None, None, 3])
# #             image = tf.image.rgb_to_grayscale(image)
# #                 image = pipeline.ratio_preserving_resize(image,
# #                                                          **config['preprocessing'])
# #             s = max(self.sizer /image.shape[:2])
# #             image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#
#             return tf.to_float(image)


        def _resize_image_cv(image):
            sizer = np.array(config['preprocessing']['resize'])
#             image = cv2.imread(path.decode('utf-8'))
#             image = cv2.imread(path.decode('utf-8'))
            print("image pre-processing: ", image.shape)

            s = max(sizer /image.shape[:2])
            print("s: ", s)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            print("image convert gray: ", image.shape)
            image = image[:int(sizer[0]/s),:int(sizer[1]/s)]
            image = cv2.resize(image, (sizer[1], sizer[0]),
                                     interpolation=cv2.INTER_AREA)
#             image = image.astype('float32') / 255.0
            print("image post-processing: ", image.shape)
            return image[:,:,np.newaxis]

        def _read_image(path):
            return cv2.imread(path.decode('utf-8'))

        def _read_image_preprocess(path):
            image = cv2.imread(path.decode('utf-8'))
            image = _resize_image_cv(image)
            return image


        def _preprocess(image):
            # tf.Tensor.set_shape(image, [None, None, 3])
            # image = tf.image.rgb_to_grayscale(image)
            # if config['preprocessing']['resize']:
            #     image = pipeline.ratio_preserving_resize(image,
            #                                              **config['preprocessing'])

            # image = _resize_image_cv(image)

            return tf.to_float(image)

        def _warp_image(image):
            H = sample_homography(tf.shape(image)[:2])
            warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
            return {'warped_im': warped_im, 'H': H}

#         def _adapt_homography_to_preprocessing(zip_data):
#             image = zip_data['image']
#             H = tf.cast(zip_data['homography'], tf.float32)
#             target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
#             s = tf.reduce_max(tf.cast(tf.divide(target_size,
#                                                 tf.shape(image)[:2]), tf.float32))
#             down_scale = tf.diag(tf.stack([1/s, 1/s, tf.constant(1.)]))
#             up_scale = tf.diag(tf.stack([s, s, tf.constant(1.)]))
#             H = tf.matmul(up_scale, tf.matmul(H, down_scale))
#             return H

        def _adapt_homography_to_preprocessing(zip_data):
            image = zip_data['image']
            H = tf.cast(zip_data['homography'], tf.float32)
            target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
            s = tf.reduce_max(tf.cast(tf.divide(target_size,
                                                tf.shape(image)[:2]), tf.float32))
#             s = tf.maximum(target_size/image.shape[:2])
            print("scale: ", s)
#             mat =tf.convert_to_tensor(np.array([[1,1,s], [1,1,s], [1/s,1/s,1]]))
            down_scale = tf.diag(tf.stack([1/s, 1/s, tf.constant(1.)]))
            up_scale = tf.diag(tf.stack([s, s, tf.constant(1.)]))
            H = tf.matmul(up_scale, tf.matmul(H, down_scale))
#             return H*mat
            return H
        
        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        homographies = tf.data.Dataset.from_tensor_slices(np.array(files['homography']))
        ## original images
        # homographies = tf.data.Dataset.zip({'image': images,
        #                                         'homography': homographies})

        # images_ori = images.copy()


        images_ori = images.map(lambda path: tf.py_func(_read_image, [path], tf.uint8))
        images = images.map(lambda path: tf.py_func(_read_image_preprocess, [path], tf.uint8))
        print("config['preprocessing']['resize']", config['preprocessing']['resize'])
        
        ##### check #####
        ### change the order, homography before process images

#         if config['preprocessing']['resize']:
        print("process homography!")
        homographies = tf.data.Dataset.zip({'image': images_ori,
                                                'homography': homographies})
        homographies = homographies.map(_adapt_homography_to_preprocessing)


        images = images.map(_preprocess)
        warped_images = tf.data.Dataset.from_tensor_slices(files['warped_image_paths'])
        warped_images = warped_images.map(lambda path: tf.py_func(_read_image_preprocess,
                                                                  [path],
                                                                  tf.uint8))
        warped_images = warped_images.map(_preprocess)

        data = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
                                    'homography': homographies})
        _ = None
        return data
