from collections import namedtuple
import cv2

Circle = namedtuple("Circle", ["x", "y", "r"])


class Rectangle:

    def __init__(self, x_l, y_b, x_r, y_t, label="Rectangle"):
        self.x_l = int(x_l)
        self.y_b = int(y_b)
        self.x_r = int(x_r)
        self.y_t = int(y_t)
        self.label = label

    @classmethod
    def from_xywh(cls, center, width, height, label="Rectangle"):
        return cls(
            x_l = int(center[0] - width / 2),
            y_b = int(center[1] - height / 2),
            x_r = int(center[0] + width / 2),
            y_t = int(center[1] + height / 2),
            label = label
        )

    def __str__(self):
        return f"Rectangle(x_l={self.x_l:.2f}, y_b={self.y_b:.2f}, x_r={self.x_r:.2f}, y_t={self.y_t:.2f}, label={self.label})"

    def __repr__(self):
        return self.__str__()

    @property
    def width(self):
        return self.x_r - self.x_l

    @property
    def height(self):
        return self.y_t - self.y_b

    @property
    def center(self):
        return (self.x_l + self.x_r) / 2, (self.y_b + self.y_t) / 2
    
    @property
    def area(self):
        return self.width * self.height

    def intersection(self, other, margin=0):
        x_l, y_b, x_r, y_t = (
            max(self.x_l, other.x_l),
            max(self.y_b, other.y_b),
            min(self.x_r, other.x_r),
            min(self.y_t, other.y_t),
        )
        if x_l < x_r + margin and y_b < y_t + margin:
            return type(self)(x_l, y_b, x_r, y_t)
        else:
            return None

    def union(self, other):
        return type(self)(
            min(self.x_l, other.x_l),
            min(self.y_b, other.y_b),
            max(self.x_r, other.x_r),
            max(self.y_t, other.y_t),
        )
    
    def iou(self, other):
        """ Intersection over Union """
        if (intersection := self.intersection(other)) is None:
            return 0
        else:       
            return intersection.area / self.union(other).area

    def __and__(self, other):
        """ intersection """
        return self.intersection(other)

    def __or__(self, other):
        """ union """
        return self.union(other)
    
    def __mul__(self, other):
        """ Resizing """
        if isinstance(other, (int, float)):
            return Rectangle.from_xywh(self.center, self.width * other, self.height * other)
        else:
            raise ValueError("Multiplication is only supported with scalar values.")

    def draw(self, image, color=(0, 0, 255), thickness=2):
        """ Draw the rectangle on the image """
        cv2.rectangle(
            image, (int(self.x_l), int(self.y_b)), (int(self.x_r), int(self.y_t)), color, thickness
        )

    def to_yolo_string(self, image_size, class_id=0):
        """ Convert rectangle to YOLO format string """

        center_x = (self.x_l + self.x_r) / 2 / image_size[1]
        center_y = (self.y_b + self.y_t) / 2 / image_size[0]
        width = self.width / image_size[1]
        height = self.height / image_size[0]
        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"