from collections import namedtuple
import torch
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import numpy as np

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

LABELS = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

def map_id_to_train_id(label_id):
    """map the id to the train id for cityscapes masks
    input: Tensor of shape (batch_size, height, width) with values from 0 to 33
    output: Tensor of shape (batch_size, height, width) with values from 0 to 18
    """
    label_id = label_id * 255
    # create a tensor with the same shape as the input tensor and fill it with the value 255
    train_id_tensor = torch.full_like(label_id, 255)
    for label in LABELS:
        # replace the value in the tensor with the train id if the value in the input tensor is equal to the id of the label
        train_id_tensor[label_id == label.id] = label.trainId
        
    return train_id_tensor

def add_gaussian_noise(image, mean=0, std=10):
    img = np.array(image)
    noise = np.random.normal(mean, std, img.shape)
    img = np.clip(img + noise, 0, 255)  
    noisy_image = Image.fromarray(np.uint8(img))
    return noisy_image

def random_brightness(image):
    factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Brightness(image)
    img_enhanced = enhancer.enhance(factor)
    return img_enhanced

def add_snow(image):
    img = image.copy()
    width, height = img.size
    overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(10000):
        x, y = random.randint(0, width), random.randint(0, height)
        d = random.randint(3, 3)
        shape = [(x, y), (x + d, y + d)]
        draw.ellipse(shape, fill=(255, 255, 255, random.randint(100, 200))) 
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    snowy_image = Image.alpha_composite(img.convert('RGBA'), overlay)
    return snowy_image.convert('RGB')

def add_fog(image):
    img = image.copy()
    width, height = img.size
    fog = Image.new('RGBA', (width, height), (255, 255, 255, int(255 * 0.6))) 
    img = Image.alpha_composite(img.convert('RGBA'), fog)
    return img.convert('RGB')

def add_rain(image):
    img = image.copy()
    draw = ImageDraw.Draw(img, 'RGBA')
    width, height = img.size 
    for _ in range(int(width * height / 100)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        end_y = y + 10
        draw.line((x, y, x, end_y), fill=(200, 200, 200, 150), width=1)
    return img

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, cityscapes_dataset, augmentation):
        self.cityscapes_dataset = cityscapes_dataset    
        self.augmentation = augmentation

    def transform(self, image, mask):
        # Resize
        original_image = image.copy()
        resize_image = transforms.Resize((256, 512), transforms.InterpolationMode.LANCZOS)
        resize_mask = transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST)
        image = resize_image(image)
        mask = resize_mask(mask)

        if self.augmentation == True:
            if random.random() < 0.25:
                # Add noise to image
                image = add_gaussian_noise(image)

            # Add brightness
            image = random_brightness(image)
            value = random.random()
            if value < 0.1:
                image = add_snow(image)
            if value > 0.1 and value < 0.2:
                image = add_fog(image)
            if value > 0.2 and value < 0.3:
                image = add_rain(image)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Normalize for the input image only
        image = TF.normalize(image, mean=[0.2870, 0.3253, 0.2840], std=[1.0, 1.0, 1.0])

        # Map train ID's
        mask = map_id_to_train_id(mask)

        return image, mask

    def __getitem__(self, index):
        image, mask = self.cityscapes_dataset[index]

        # Directly use the transform method to apply the same transforms on both image and mask
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.cityscapes_dataset)