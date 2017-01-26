from resnet50 import ResNet50

model = ResNet50(include_top=True, weights='imagenet')
model = ResNet50(include_top=False, weights='imagenet')
