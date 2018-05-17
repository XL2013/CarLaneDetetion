import os
import json
import numpy as np
from PIL import Image, ImageDraw
import imgaug


from mrcnn import model as modellib, utils
from mrcnn.config import Config




DEFAULT_LOGS_DIR = '../log'
COCO_WEIGHTS_PATH = '../saved_model/mask_rcnn_coco.h5'


############################################################
#  Configurations
############################################################


class CarlaneConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "carlane"
    
    BACKBONE = "resnet50"
    #BACKBONE = "resnet101"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    
    # NUMBER OF GPUS to use
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21  # Background + 21 carlane classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 400

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped    
    DETECTION_MIN_CONFIDENCE = 0.7
    
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    
    # Image mean(RGB)
    MEAB_PIXEL = np.array([115.5, 115.9, 116.2])


############################################################
#  Dataset
############################################################

class CarlaneDataset(utils.Dataset):


    def load_carlane(self, dataset_dir, subset):
        """Load a subset of the Carlane dataset
        dataset_dir: Root directory of the dataset
        subset: Subset to load: train or val

        """ 
        assert subset in ["train", "val"]
        
        # train_val.json:{
        #    "train": xx,
        #    "val":xx,
        #    "classes":xx
        #    }
        # train, val: [image_name,...]
        # classes: [(class_name, class_id),...]
       
        data = json.load(
            open(os.path.join(dataset_dir, 'train_val.json'), 'r'))

        classes = data['classes']
        for c in classes:
            self.add_class("carlane", c[1], c[0])

        self.class_map = {}
        for c in classes:
            self.class_map[c[0]] = c[1]

        image_names = data[subset]

        # Add images and Load annotaions
        for image_name in image_names:
            json_path = os.path.join(dataset_dir, image_name + '.json')
            annotation = json.load(open(json_path))
            annotation = annotation['datalist']
            polygons = [a['arr'] for a in annotation]
            polygons = [[(xy['x'], xy['y']) for xy in p] for p in polygons]
            polygon_types = [a['type'] for a in annotation]

            image_path = os.path.join(dataset_dir, image_name + '.jpg')
            image = Image.open(image_path)
            width, height = image.size

            self.add_image(
                "carlane",
                image_id=image_name,
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                polygon_types=polygon_types)

    def load_mask(self, image_id):
        """Generate instance masks for an image
        
        Returns:
         masks:  A bool array of shape [height, width, instance count] with
             one mask per instance
         polygon_ids: a 1d array of class IDs of the instance masks
        """
                
        # Convert polygons to a bitmap mask of shape
        # [height, width, insstance_count]
        info = self.image_info[image_id]
        polygons = info["polygons"]
        polygon_types = info["polygon_types"]
        mask = np.zeros([info["height"], info["width"], len(polygons)], dtype=np.uint8)
        for i, polygon in enumerate(polygons):
            m = Image.new('L', (info["width"], info["height"]), color=0)
            draw = ImageDraw.Draw(m)
            draw.polygon(polygon, fill=1, outline=1)
            mask[:, :, i] = np.array(m)
        polygon_ids = np.array([self.class_map[t] for t in polygon_types])
        return mask.astype(np.bool), polygon_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image"""
        info = self.image_info[image_id]
        if info["source"] == "carlane":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Lane line')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Lane Line")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/lane line/",
                        help='Directory of the Lane Line dataset')

    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco' or 'imagenet' ")


    args = parser.parse_args()
    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Weights:", args.weights)
    if args.command == "train":
        config = CarlaneConfig()
    else:
        class InferenceConfig(CarlaneConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

      # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
        

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
   
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
       # Training dataset.
        dataset_train = CarlaneDataset()
        dataset_train.load_carlane(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CarlaneDataset()
        dataset_val.load_carlane(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***
        
        if args.weights.lower() == 'imagenet':
            # Training - Stage 1
            print("Training network heads")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=40,
                        layers='heads',
                        augmentation=augmentation)

        '''
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=80,
                    layers='4+',
                    augmentation=augmentation)
        '''
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # TODO
        pass
        
            
            
            
           