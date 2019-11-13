import os
import shutil
import glob

cats_names = glob.glob("train/cat*.jpg")
dogs_names = glob.glob("train/dog*.jpg")

for i,name in enumerate(cats_names):
    name2 = "train/cats/"+str(i)+".jpg"
    print("moving cat ",i)
    shutil.move(name,name2)

for i,name in enumerate(dogs_names):
    name2 = "train/dogs/"+str(i)+".jpg"
    print("moving dog ",i)
    shutil.move(name,name2)


print("done")
