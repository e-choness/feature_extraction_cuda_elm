import os
from PIL import Image

root_path = 'E:/PycharmProjects/pythonProject/datasets/Caltech256/'
new_root = './Caltech256_new/'
folders = sorted(os.listdir(root_path))
train_number = 30
train_folder = 'train/'
test_folder = 'test/'
train_path = os.path.join(new_root, train_folder)
test_path = os.path.join(new_root, test_folder)
print(train_path)
print(test_path)
categories_number = 0
total = 0

for folder in folders:
    folder = folder + '/'
    images_path = os.path.join(root_path, folder)
    images = sorted(os.listdir(images_path))
    counter = 0
    train_counter = 0
    test_counter = 0
    #print(images)
    for image in images:
        categories_train = os.path.join(train_path, folder)
        if (os.path.exists(categories_train) == False):
            os.makedirs(categories_train)
        categories_test = os.path.join(test_path, folder)
        if (os.path.exists(categories_test) == False):
            os.makedirs(categories_test)
        img_complete_path = os.path.join(images_path, image)
        temp = Image.open(img_complete_path)
        # print(img_complete_path)
        if (counter < train_number):
            image_train = os.path.join(categories_train,image)
            temp.save(image_train)
            print(image_train)
            train_counter += 1
        else:
            image_test = os.path.join(categories_test, image)
            temp.save(image_test)
            print(image_test)
            test_counter += 1
        counter += 1
    print('train', train_counter)
    print('test', test_counter)
    print(categories_test)
    print(categories_train)

    categories_number += 1
    print('current category', categories_number)

    total += counter

print(total, " images in ", categories_number, ' classes')

