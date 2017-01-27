# ResNet 50 inside Docker
Easily extract image features from various layers of ResNet50.
Or just use it in prediction mode to get labels for input images.

## Tests

To see if you are using the weights correctly, check out `/resnet_50/model/model_test.py`.
It will predict the top5 class labels for each `/resnet_50/model/test_images/*.jpg`.
This script is run during the Docker image build to verify predictions are reasonable.

### Sources of test images:
- cat1.jpg: [Dwight Sipler](http://www.flickr.com/people/62528187@N00) from Stow, MA, USA, [Gillie hunting (2292639848)](https://commons.wikimedia.org/wiki/File:Gillie_hunting_(2292639848).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/legalcode)
- cat2.jpg: The original uploader was [DrL](https://en.wikipedia.org/wiki/User:DrL) at [English Wikipedia](https://en.wikipedia.org/wiki/) [Blackcat-Lilith](https://commons.wikimedia.org/wiki/File:Blackcat-Lilith.jpg), [CC BY-SA 2.5
](https://creativecommons.org/licenses/by-sa/2.5/legalcode)
- dog1.jpg: HiSa Hiller, Schweiz, [Thai-Ridgeback](https://commons.wikimedia.org/wiki/File:Thai-Ridgeback.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode)
- dog2.jpg: [Military dog barking](https://commons.wikimedia.org/wiki/File:Military_dog_barking.JPG), in the [public domain](https://en.wikipedia.org/wiki/public_domain)
- ipod.jpg: [Marcus Quigmire](http://www.flickr.com/people/41896843@N00) from Florida, USA, [Baby Bloo taking a dip (3402460462)](https://commons.wikimedia.org/wiki/File:Baby_Bloo_taking_a_dip_(3402460462).jpg), [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/legalcode)
