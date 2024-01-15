import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

files = []

#set raw folder
raw_dir = 'my-pets'
#set final folder
final_dir = 'preprocessing-imgs\\'
#set img size
size = (299, 299)

#verify a final folder for create if necessary
if not os.path.exists(final_dir):
    os.makedirs(final_dir)
    print("create a new folder: ", final_dir)

#run raw files
for root_path, _, arquivos_na_pasta in os.walk(raw_dir):
    for file in arquivos_na_pasta:
        dir_files = os.path.join(root_path, file)
        files.append(dir_files)

#resize files
for file in files:
    print("converter file:", file.replace(raw_dir, final_dir))
    img = load_img(file, target_size=size)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))

    #verify a class folder for create if necessary
    if not os.path.exists(final_dir + file.split('\\')[1]):
        os.makedirs(final_dir + file.split('\\')[1])
        print("create a new folder: ", final_dir + file.split('\\')[1])

    img_converted = os.path.join(file.replace(raw_dir, final_dir))
    #save a img converted
    img.save(img_converted)

print("pre-processing images done!")