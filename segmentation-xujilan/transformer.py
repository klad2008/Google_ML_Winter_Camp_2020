import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageOps
import collections
from matplotlib import pyplot as plt


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL = True

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            self.PIL = False

        if self.PIL:
            img = np.array(img)
            mask = np.array(mask)

        assert img.shape[:2] == mask.shape[:2]

        for a in self.augmentations:
            img, mask = a(img, mask)

        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        # assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)),
            mask,
        )


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask


class CenterCrop(object):
    def __init__(self, size, padding=0):
        self.padding = padding
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST))

        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))

'''
class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, contour):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), contour.transpose(Image.FLIP_LEFT_RIGHT))
        return img, mask, contour


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM))
        return img, mask
'''


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label




class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST))


class RandomTranslate(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(
            img,
            y_crop_offset,
            x_crop_offset,
            img.size[1] - abs(y_offset),
            img.size[0] - abs(x_offset),
        )

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
            tf.pad(cropped_img, padding_tuple, padding_mode="reflect"),
            tf.affine(
                mask,
                translate=(-x_offset, -y_offset),
                scale=1.0,
                angle=0.0,
                shear=0.0,
                fillcolor=250,
            ),
        )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=(0, 0, 0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=250,
                shear=0.0,
            ),
        )


class Scale(object):
    def __init__(self, size_w, size_h):
        self.size_w = size_w
        self.size_h = size_h

    def __call__(self, img, mask):
        assert img.shape[:2] == mask.shape[:2]
        w, h = img.shape[:2]
        if (w >= h and w == self.size_w) or (h >= w and h == self.size_h):
            return img, mask
        if w > h:
            ow = self.size_w
            oh = int(self.size_h * h / w)
            return cv2.resize(img, (ow, oh)), cv2.resize(mask, (ow, oh), cv2.INTER_NEAREST)
        else:
            ow = int(self.size_w * w / h)
            oh = self.size_h
            return cv2.resize(img, (ow, oh)), cv2.resize(mask, (ow, oh), cv2.INTER_NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST))

        return self.crop(*self.scale(img, mask))


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label



class Sliding_Window_Crop(object):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = mask.size

        if w == self.size and h == self.size:
            return [Image.fromarray(img)], [Image.fromarray(mask)]
        if w < self.size or h < self.size:
            img = img.resize((max(w, self.size), max(h, self.size)), Image.BILINEAR)
            mask = mask.resize((max(w, self.size), max(h, self.size)), Image.NEAREST)

            # return [img.resize((self.size, self.size), Image.BILINEAR)], \
            #        [mask.resize((self.size, self.size), Image.NEAREST)]

        img = np.array(img)
        mask = np.array(mask)

        # Notice we swap to match PIL image with numpy array shape
        w, h = h, w

        step_w = int(np.ceil((w - self.size) / self.stride)) + 1
        step_h = int(np.ceil((h - self.size) / self.stride)) + 1

        img_list = []
        mask_list = []

        counter = 0
        for i in range(step_w):
            for j in range(step_h):
                start_x = i * self.stride
                start_y = j * self.stride
                end_x = start_x + self.size
                end_y = start_y + self.size
                if end_x >= w:
                    end_x = w
                    start_x = end_x - self.size
                if end_y >= h:
                    end_y = h
                    start_y = end_y - self.size

                img_crop = img[start_x:end_x, start_y:end_y, :]
                mask_crop = mask[start_x:end_x, start_y:end_y]

                img_list.append(Image.fromarray(img_crop))
                mask_list.append(Image.fromarray(mask_crop))

                # img_crop = img.crop((start_x, start_y, end_x, end_y))
                # mask_crop = mask.crop((start_x, start_y, end_x, end_y))
                # img_list.append(img_crop)
                # mask_list.append(mask_crop)

                # temp_dir = '/home/xujl/data/results/frrn_debug/'
                # img_temp =Image.fromarray(img_crop)
                # img_temp.save(temp_dir + str(counter) + '.png')
                # counter = counter + 1


                # print(start_x, end_x, start_y, end_y, img_crop.size)
        return img_list, mask_list


class SW_Merge_Prediction(object):
    def __init__(self, size, stride, original_w, original_h):
        self.size = size
        self.stride = stride
        self.original_w = original_w
        self.original_h = original_h

    def __call__(self, mask_list):
        pred = np.array(np.zeros([self.original_w, self.original_h])).astype(np.float64)
        counter = pred + (1 + np.finfo(np.float32).eps)

        total = len(mask_list)
        step_w = int(np.ceil((self.original_w - self.size) / self.stride))
        step_h = int(np.ceil((self.original_h - self.size) / self.stride))

        idx = 0
        for i in range(step_w):
            for j in range(step_h):
                start_x = i * self.stride
                start_y = j * self.stride
                end_x = start_x + self.size
                end_y = start_y + self.size

                if i == step_w - 1:
                    end_x = self.original_w
                    start_x = end_x - self.size

                if j == step_h - 1:
                    end_y = self.original_h
                    start_y = end_y - self.size

                pred[start_x:end_x, start_y:end_y] += mask_list[idx]
                counter[start_x:end_x, start_y:end_y] += 1
                idx += 1

        pred /= counter
        pred = pred[np.newaxis, :, :]
        return pred


class SWA_Merge_Prediction(object):
    def __init__(self, size, stride, original_w, original_h):
        self.size = size
        self.stride = stride
        self.original_w = original_w
        self.original_h = original_h
        self.weight, self.ori_weight = self.get_weight()
        self.gauss = self.get_gauss(window_size=size, sigma=1.5)

    def get_gauss(self, window_size, sigma=1.5):
        one_gauss = np.array(
            [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        one_gauss /= np.sum(one_gauss)

        two_gauss = np.matmul(one_gauss, one_gauss.T)
        return two_gauss

    def get_weight(self):
        weight = np.zeros([self.size, self.size])
        for i in range(self.size):
            for j in range(self.size):
                dx = min(i, self.size - i)
                dy = min(j, self.size - j)
                d = min(dx, dy)
                weight[i, j] = d + 1

        weight /= weight.max()

        ori_weight = np.zeros([self.original_w, self.original_h])
        for i in range(self.original_w):
            for j in range(self.original_h):
                dx = min(i, self.original_w - i)
                dy = min(j, self.original_h - j)
                d = min(dx, dy)
                ori_weight[i, j] = d + 1

        ori_weight /= ori_weight.max()

        return weight, ori_weight

    def __call__(self, mask_list):
        ow = max(self.original_w, self.size)
        oh = max(self.original_h, self.size)

        pred = np.array(np.zeros([ow, oh])).astype(np.float64)
        counter = pred + 1

        total = len(mask_list)
        step_w = int(np.ceil((ow - self.size) / self.stride)) + 1
        step_h = int(np.ceil((oh - self.size) / self.stride)) + 1

        # print('step_w, ', step_w, 'step_h ', step_h, 'total: ', total)
        # print('self.size ', self.size, 'self.stride, ', self.stride, 'ow, ', self.original_w, 'oh', self.original_h)
        idx = 0

        for i in range(step_w):
            for j in range(step_h):
                start_x = i * self.stride
                start_y = j * self.stride
                end_x = start_x + self.size
                end_y = start_y + self.size

                if end_x >= ow:
                    end_x = ow
                    start_x = end_x - self.size

                if end_y >= oh:
                    end_y = oh
                    start_y = end_y - self.size


                # pred[start_x:end_x, start_y:end_y] += mask_list[idx] * self.gauss
                # counter[start_x:end_x, start_y:end_y] += self.gauss
                pred[start_x:end_x, start_y:end_y] += mask_list[idx]
                counter[start_x:end_x, start_y:end_y] += 1
                idx += 1

        pred /= counter
        # pred /= self.ori_weight
        pred = pred[0:self.original_w, 0:self.original_h]
        pred = pred[np.newaxis, :, :]
        return pred


class elastic_transform(object):
    def __init__(self):
        pass

    def __call__(self, img, mask):
        img = np.array(img)
        mask = np.array(mask)

        image = np.concatenate((img, mask[:, :, np.newaxis]), axis=2)

        alpha = img.shape[0] * 2
        sigma = img.shape[0] * 0.08
        alpha_affine = img.shape[0] * 0.08
        random_state = None

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image, M, shape_size, borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

        img = image[:, :, :3]
        mask = image[:, :, 3]
        return img, mask


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.05, 1.5), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio


    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)

        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return tf.resized_crop(img, i, j, h, w, self.size, self.interpolation),\
               tf.resized_crop(mask, i, j, h, w, self.size, self.interpolation),\


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, mask):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, mask


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='rand', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        #print(label.shape)
        h, w = label.shape[:2]
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape[:2]
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        # assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, mask):
        return self.lambd(img), self.lambd(mask)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: tf.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: tf.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: tf.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: tf.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img, mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

'''
if __name__ == '__main__':

    elastic = elastic_transform()
    img = Image.open('/home/xujl/data/gland/train/train_1.bmp')
    mask = Image.open('/home/xujl/data/gland/train_masks/train_anno_1.bmp')
    contour = Image.open('/home/xujl/data/gland/train_contour/train_anno_1.bmp')

    img2, mask2, contour2 = elastic(img, mask, contour)
    # img2 = elastic(img, mask, contour)

    plt.imshow(img)
    plt.show()

    plt.imshow(img2)
    plt.show()


    plt.imshow(np.array(mask))
    plt.show()


    plt.imshow(np.array(mask2))
    plt.show()

    plt.imshow(np.array(contour))
    plt.show()

    plt.imshow(np.array(contour2))
    plt.show()
'''

