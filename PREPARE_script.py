import os
path_root = "F:\COLLEGE\Computer Vision\PROJECT\SignatureObjectDetection"

training_images_path = path_root + '\\TestGroundTruth'
training_images_path_2 = path_root + '\\TestGroundTruth_2'


os.mkdir(training_images_path_2)

for i in os.listdir(training_images_path):
    f = open(training_images_path + "\\" + i , 'r')


    stringg = f.readlines()

    for j in range(0,len(stringg)):
        if j == 0:
            res= []
            readd = stringg[j].split(",")


            xmin = int(readd[0])
            ymin = int(readd[1])
            xmax = int(readd[2])
            ymax = int(readd[3])

            # Transform the bbox co-ordinates as per the format required by YOLO v5
            b_center_x = (xmin + xmax) / 2
            b_center_y = (ymin + ymax) / 2
            b_width = (xmax - xmin)
            b_height = (ymax - ymin)

            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, image_c = [1000 , 1000 , 3]
            b_center_x /= image_w
            b_center_y /= image_h
            b_width /= image_w
            b_height /= image_h

            classs = 0

            f= open(training_images_path_2 + "\\" + i , 'w+')

            f.write(str(classs) + " " + str(b_center_x) + " " + str(b_center_y) + " " + str(b_width) + " " + str(b_height) + "\n")
        else:
            res = []
            readd = stringg[j].split(",")

            xmin = int(readd[0])
            ymin = int(readd[1])
            xmax = int(readd[2])
            ymax = int(readd[3])

            # Transform the bbox co-ordinates as per the format required by YOLO v5
            b_center_x = (xmin + xmax) / 2
            b_center_y = (ymin + ymax) / 2
            b_width = (xmax - xmin)
            b_height = (ymax - ymin)

            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, image_c = [1000, 1000, 3]
            b_center_x /= image_w
            b_center_y /= image_h
            b_width /= image_w
            b_height /= image_h

            classs = 0

            f.write(str(classs) + " " + str(b_center_x) + " " + str(b_center_y) + " " + str(b_width) + " " + str(b_height) + "\n")

    f.close()


