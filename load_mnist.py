import numpy as np 
import struct
import os

test_label_path = 'C:\\Users\\somak\\Documents\\somak_python\\deep_learning\\mnist\\test_set\\labels\\'
test_images_path = 'C:\\Users\\somak\\Documents\\somak_python\\deep_learning\\mnist\\test_set\\images\\'
train_label_path = 'C:\\Users\\somak\\Documents\\somak_python\\deep_learning\\mnist\\training_set\\labels\\'
train_images_path = 'C:\\Users\\somak\\Documents\\somak_python\\deep_learning\\mnist\\training_set\\images\\'
pickled_form = 'C:\\Users\\somak\\Documents\\somak_python\\deep_learning\\mnist\\minst_compressed.npz'

'''

## from the page: http://yann.lecun.com/exdb/mnist/

##   start:
## The data is stored like in a C array, i.e. the index in the last dimension changes the fastest. 
## All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. 
## Users of Intel processors and other low-endian machines must flip the bytes of the header.
##   end:

struct is a way of converting between python and c structs represented in python byte objects

eg:

>>> struct.pack('iii',12,15,16)
b'\x0c\x00\x00\x00\x0f\x00\x00\x00\x10\x00\x00\x00'
>>> aa = struct.pack('iii',12,15,16)
>>> aa
b'\x0c\x00\x00\x00\x0f\x00\x00\x00\x10\x00\x00\x00'
>>> struct.unpack('III', aa)
(12, 15, 16)

'''

def load_labels(path = train_label_path, kind = 'train'):
    """

        from mnist database page (http://yann.lecun.com/exdb/mnist/): \n
        
        The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
        
        TRAINING/TEST SET LABEL FILE (train-labels-idx1-ubyte):
        [offset] [type]          [value]          [description] 
        0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
        0004     32 bit integer  60000            number of items 
        0008     unsigned byte   ??               label 
        0009     unsigned byte   ??               label 
        .....\n
        .....


        the above means that actual data starts from the 8th element. So we read the magic number, number of items
        using struct and then use the numpy library to convert the remaining labels to numpy matrix


        how to read the magic number and number of labels? -- struct.unpack('>II', lbl.read(8))
        ??why > -- this denotes high endian
        ?? why II -- we are reading 2 values in int mode, hence I**2
        

    """

    label_path = os.path.join(path, '%s-labels.idx1-ubyte'%kind)

    with open(label_path, 'rb') as lbl:
        magic, number_of_labels = struct.unpack('>II', lbl.read(8))
        labels = np.fromfile(lbl, dtype = np.uint8)

    print(len(labels), number_of_labels)
    return labels

def load_images(path = train_images_path, kind = 'train', number_of_labels = 55000) :
    """

        from mnist database page (http://yann.lecun.com/exdb/mnist/): \n
        
        The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
        
        TRAINING/TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
        [offset] [type]          [value]          [description] 
        0000     32 bit integer  0x00000803(2051) magic number 
        0004     32 bit integer  10000            number of images 
        0008     32 bit integer  28               number of rows 
        0012     32 bit integer  28               number of columns 
        0016     unsigned byte   ??               pixel 
        0017     unsigned byte   ??               pixel 
        ........ 
        xxxx     unsigned byte   ??               pixel
        .....\n
        .....


        the above means that actual data starts from the 16th element. So we read the magic number, number of items
        , number of rows, and number of columns using struct and then use the numpy library to convert the remaining 
        labels to numpy matrix


        how to read the magic number and number of labels? -- struct.unpack('>IIII', lbl.read(16))
        ??why > -- this denotes high endian
        ?? why II -- we are reading 4 values in int mode, hence I**4

        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
    """

    images_path = os.path.join(path, '%s-images.idx3-ubyte'%kind)

    with open(images_path, 'rb') as img:

        magic, number_of_images, rows, cols = struct.unpack('>IIII', img.read(16))

        images = np.fromfile(img, dtype = np.uint8).reshape(number_of_labels, 784) ## why 784, bcoz its a 28*28 matrix

        images = ((images/255) - 0.5) *2 ## scaling the values within 0 and 1 (originally they are between 0 nad 255)

    print('Number of images %s, number of rows %s number of cols %s' %(number_of_images, rows, cols))
    return images

def pickle_to_drive(x_train, x_test, y_train, y_test, path = pickled_form):
    
    np.savez_compressed(path, x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

def load_from_pickled_form(path = pickled_form):
    return np.load(path)

test_labels = load_labels(path = test_label_path, kind = 't10k') ## t10k is the test mnemonic
test_images = load_images(path = test_images_path, kind = 't10k', number_of_labels = len(test_labels))
train_labels = load_labels()
train_images = load_images(number_of_labels = len(train_labels))

pickle_to_drive(train_images, test_images, train_labels, test_labels)




