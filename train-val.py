import os

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # classes = ["jk", "kiss", "other", "skirt", "stockings"]
    classes = ["0", "1", "2", "3", "4", "5", "6", "7"]
    
    path = "video_label_data"
    # path = "video_label_data"
    label_list = os.listdir(path)

    
    x_train, x_test = [], []
    y_train, y_test = [], []
    for label in label_list:
        label_index = classes.index(label)
        label_path = os.path.join(path, label)
        image_list = os.listdir(label_path)
        
        y_ = []
        x_ = []
        for image in image_list:
            image_path = os.path.join(label_path, image)
            x_.append(image_path)
            y_.append(label_index)
            
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(x_, y_, test_size=0.25)

        x_train.extend(X_train_e)
        x_test.extend(X_test_e)
        y_train.extend(y_train_e)
        y_test.extend(y_test_e)
    print(len(x_train), len(y_train), len(x_test), len(y_test))

    f_train = open('train.txt', mode='w', encoding="utf-8")
    f_test = open('test.txt', mode='w', encoding="utf-8")
    
    for i in range(len(x_train)):
        line = x_train[i] + " " + str(y_train[i]) + "\n"
        f_train.write(line)
    f_train.close()

    for i in range(len(x_test)):
        line = x_test[i] + " " + str(y_test[i]) + "\n"
        f_test.write(line)
    f_test.close()


    



