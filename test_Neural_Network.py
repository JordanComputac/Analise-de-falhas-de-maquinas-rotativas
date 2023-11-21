import FunctionModule as fm

y_true_classes=[[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
y_pred_classes=[[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]

confusion = fm.confusion_matrix(y_true_classes, y_pred_classes)
for row in confusion:
    print(row)

for i in range(4):  # Loop from 1 to 4 (inclusive)
    print(i)


# conf_matrix = [[10, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 12, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 6, 3, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0], [0, 0, 6, 0, 0, 0, 0, 0, 12, 0, 0, 0, 5, 0, 5, 1, 0, 0, 0, 0, 0, 7, 0, 0], [0, 0, 0, 11, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 10, 0, 0], [0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 13, 0, 4, 0, 0, 0, 6, 0, 0, 0, 0, 0, 12, 0], [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 13, 9, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 6], [8, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 8, 0, 3, 0, 0], [0, 0, 14, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 15, 0, 0, 0, 2, 0, 0, 4, 0, 0], [0, 0, 0, 13, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 10, 0, 0], [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 13, 0, 3, 7, 0, 1, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 9, 1, 2, 0, 0, 0, 3, 2, 0, 0, 0, 4, 6], [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0], [0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 11, 0, 0, 0], [0, 0, 0, 6, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 5, 0, 0], [1, 0, 0, 1, 9, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 9, 0], [5, 0, 0, 0, 0, 7, 1, 8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 11], [14, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0], [0, 11, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0], [0, 0, 10, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 8, 0, 0, 0], [0, 0, 0, 6, 2, 0, 0, 0, 0, 9, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0], [0, 0, 0, 0, 6, 0, 0, 0, 0, 1, 10, 1, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 14, 0], [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 10]]
# conf_matrix = [[7, 0, 0, 0, 0, 0, 0, 1, 5, 0, 2, 2, 1, 1, 2, 2, 2, 3, 3, 4, 0, 0, 2, 2], [0, 12, 0, 0, 0, 0, 2, 1, 1, 2, 3, 3, 2, 1, 2, 0, 2, 2, 1, 1, 1, 1, 2, 1], [0, 0, 11, 0, 0, 0, 0, 3, 1, 2, 1, 1, 2, 3, 4, 0, 1, 1, 0, 3, 1, 1, 2, 0], [0, 0, 0, 9, 0, 0, 0, 1, 1, 1, 0, 0, 0, 4, 0, 3, 5, 0, 1, 1, 4, 3, 3, 1], [0, 0, 0, 0, 12, 0, 2, 3, 1, 1, 4, 3, 2, 2, 2, 2, 0, 1, 1, 3, 3, 2, 1, 3], [0, 0, 0, 0, 0, 9, 3, 3, 2, 3, 2, 0, 0, 1, 1, 2, 2, 2, 1, 0, 2, 2, 2, 2], [0, 2, 0, 0, 2, 3, 10, 0, 0, 0, 0, 0, 3, 4, 1, 0, 1, 2, 0, 4, 2, 0, 2, 3], [1, 1, 3, 1, 3, 3, 0, 12, 0, 0, 0, 0, 3, 2, 2, 0, 1, 1, 1, 1, 1, 0, 1, 3], [5, 1, 1, 1, 1, 2, 0, 0, 8, 0, 0, 0, 2, 2, 0, 0, 4, 3, 1, 0, 1, 0, 3, 2], [0, 2, 2, 1, 1, 3, 0, 0, 0, 3, 0, 0, 2, 1, 0, 1, 5, 3, 2, 0, 2, 1, 6, 2], [2, 3, 1, 0, 4, 2, 0, 0, 0, 0, 14, 0, 0, 1, 3, 2, 1, 2, 4, 5, 0, 2, 0, 2], [2, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 13, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 1], [1, 2, 2, 0, 2, 0, 3, 3, 2, 2, 0, 0, 11, 0, 0, 0, 0, 0, 2, 3, 1, 2, 2, 1], [1, 1, 3, 4, 2, 1, 4, 2, 2, 1, 1, 2, 0, 9, 0, 0, 0, 0, 0, 1, 2, 0, 1, 3], [2, 2, 4, 0, 2, 1, 1, 2, 0, 0, 3, 2, 0, 0, 11, 0, 0, 0, 0, 0, 1, 3, 3, 0], [2, 0, 0, 3, 2, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 12, 0, 0, 2, 2, 3, 3, 0, 3], [2, 2, 1, 5, 0, 2, 1, 1, 4, 5, 1, 2, 0, 0, 0, 0, 9, 0, 4, 2, 2, 2, 2, 1], [3, 2, 1, 0, 1, 2, 2, 1, 3, 3, 2, 2, 0, 0, 0, 0, 0, 8, 3, 1, 2, 2, 1, 0], [3, 1, 0, 1, 1, 1, 0, 1, 1, 2, 4, 2, 2, 0, 0, 2, 4, 3, 11, 0, 0, 0, 0, 0], [4, 1, 3, 1, 3, 0, 4, 1, 0, 0, 5, 2, 3, 1, 0, 2, 2, 1, 0, 7, 0, 0, 0, 0], [0, 1, 1, 4, 3, 2, 2, 1, 1, 2, 0, 2, 1, 2, 1, 3, 2, 2, 0, 0, 7, 0, 0, 0], [0, 1, 1, 3, 2, 2, 0, 0, 0, 1, 2, 0, 2, 0, 3, 3, 2, 2, 0, 0, 0, 13, 0, 0], [2, 2, 2, 3, 1, 2, 2, 1, 3, 6, 0, 2, 2, 1, 3, 0, 2, 1, 0, 0, 0, 0, 13, 0], [2, 1, 0, 1, 3, 2, 3, 3, 2, 2, 2, 1, 1, 3, 0, 3, 1, 0, 0, 0, 0, 0, 0, 9]]
# conf_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [17, 18, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 1, 10, 0, 9, 0, 0, 0, 9, 0, 6, 0, 0], [17, 7, 7, 0, 0, 0, 4, 14, 15, 0, 0, 0, 1, 0, 8, 5, 0, 2, 5, 0, 4, 9, 0, 3], [21, 0, 0, 21, 5, 0, 0, 0, 0, 35, 0, 0, 0, 6, 0, 36, 0, 0, 0, 0, 0, 39, 0, 5], [21, 1, 0, 0, 14, 1, 0, 0, 0, 0, 40, 0, 3, 19, 0, 7, 10, 5, 14, 1, 1, 3, 19, 3], [26, 1, 0, 0, 4, 15, 0, 1, 0, 1, 7, 22, 5, 13, 0, 3, 0, 14, 9, 1, 0, 6, 7, 24], [38, 0, 1, 0, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 22, 1, 0, 18, 0, 0, 40, 0, 0, 0, 0, 8, 9, 0, 3, 7, 8, 0, 23, 0, 0, 0, 0], [0, 0, 35, 0, 5, 0, 0, 0, 38, 0, 0, 0, 10, 0, 0, 4, 13, 1, 0, 0, 35, 0, 1, 0], [4, 0, 0, 23, 11, 1, 0, 0, 0, 45, 0, 0, 12, 1, 0, 15, 3, 1, 9, 0, 0, 36, 0, 0], [1, 0, 0, 0, 42, 0, 0, 0, 0, 0, 38, 0, 4, 3, 8, 0, 26, 4, 1, 0, 0, 1, 28, 11], [1, 0, 0, 0, 5, 29, 0, 0, 0, 0, 0, 49, 5, 3, 0, 0, 17, 26, 4, 1, 0, 3, 22, 11], [13, 0, 0, 15, 4, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 20, 5, 0, 0, 11, 0], [13, 6, 0, 17, 5, 0, 0, 36, 1, 0, 3, 0, 0, 52, 0, 0, 0, 0, 0, 39, 0, 0, 0, 5], [0, 0, 37, 4, 1, 0, 0, 9, 22, 0, 13, 0, 0, 0, 34, 0, 0, 0, 0, 0, 46, 0, 1, 1], [0, 0, 16, 22, 2, 0, 3, 9, 8, 16, 1, 5, 0, 0, 0, 47, 0, 0, 2, 0, 2, 9, 19, 1], [1, 0, 9, 16, 21, 0, 0, 5, 4, 0, 24, 1, 0, 0, 0, 0, 32, 0, 0, 5, 2, 1, 24, 2], [9, 0, 8, 3, 8, 10, 0, 0, 0, 0, 14, 28, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, 45], [38, 0, 0, 12, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0], [0, 11, 2, 24, 0, 0, 12, 12, 0, 1, 0, 0, 0, 32, 0, 0, 2, 7, 0, 37, 0, 0, 0, 0], [0, 0, 34, 1, 0, 0, 0, 0, 34, 0, 6, 0, 0, 0, 32, 0, 2, 3, 0, 0, 41, 0, 0, 0], [1, 2, 0, 24, 4, 1, 0, 0, 0, 34, 15, 0, 0, 1, 0, 37, 3, 6, 0, 0, 0, 44, 0, 0], [1, 0, 17, 2, 26, 0, 1, 0, 0, 15, 23, 0, 1, 6, 2, 0, 20, 5, 0, 0, 0, 2, 34, 0], [18, 0, 0, 3, 2, 17, 12, 9, 2, 3, 4, 13, 1, 8, 0, 0, 6, 18, 0, 0, 0, 0, 1, 38]]

# All accelerations and rot_freq considered (Fourier Treatment)
conf_matrix = [[32, 1, 0, 0, 0, 0, 44, 1, 0, 0, 0, 0, 29, 0, 4, 1, 8, 2, 10, 0, 8, 0, 0, 9], [0, 46, 0, 0, 0, 0, 11, 20, 3, 0, 2, 2, 0, 18, 11, 1, 1, 2, 0, 23, 15, 0, 0, 12], [0, 0, 48, 0, 0, 0, 0, 0, 39, 0, 2, 0, 3, 0, 20, 8, 1, 0, 0, 0, 38, 0, 0, 0], [0, 0, 8, 31, 1, 0, 0, 0, 1, 25, 12, 0, 3, 6, 0, 21, 0, 18, 2, 9, 5, 17, 13, 0], [0, 0, 3, 1, 27, 0, 0, 0, 0, 6, 31, 0, 3, 2, 0, 18, 12, 3, 0, 5, 13, 4, 18, 1], [0, 0, 0, 0, 0, 42, 0, 0, 4, 0, 0, 37, 4, 1, 0, 17, 0, 23, 0, 0, 1, 12, 0, 25], [26, 3, 1, 0, 7, 0, 34, 0, 0, 0, 0, 0, 5, 1, 7, 9, 3, 14, 44, 0, 0, 0, 0, 0], [0, 40, 0, 0, 6, 0, 0, 41, 0, 0, 0, 0, 0, 21, 0, 13, 2, 5, 1, 29, 2, 0, 0, 1], [0, 0, 50, 0, 0, 0, 0, 0, 32, 0, 0, 0, 1, 1, 19, 10, 5, 1, 0, 0, 36, 0, 0, 0], [0, 0, 0, 22, 8, 5, 0, 0, 0, 36, 5, 0, 6, 4, 0, 17, 7, 3, 0, 1, 0, 27, 7, 0], [0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 53, 1, 3, 3, 6, 3, 30, 2, 1, 2, 2, 7, 29, 0], [4, 0, 0, 2, 5, 27, 0, 0, 0, 0, 0, 38, 1, 1, 2, 8, 15, 12, 9, 11, 0, 7, 0, 24], [23, 0, 0, 0, 20, 0, 25, 11, 1, 10, 0, 0, 37, 0, 0, 0, 0, 0, 28, 11, 0, 0, 0, 0], [26, 2, 0, 0, 16, 0, 3, 11, 0, 8, 0, 4, 0, 39, 0, 0, 0, 0, 11, 32, 0, 0, 0, 1], [12, 1, 21, 1, 1, 0, 6, 2, 31, 0, 0, 1, 0, 0, 47, 0, 0, 0, 0, 0, 25, 5, 3, 0], [12, 2, 0, 14, 5, 0, 0, 3, 1, 29, 8, 0, 0, 0, 0, 44, 0, 0, 0, 0, 4, 19, 14, 2], [23, 1, 0, 4, 15, 2, 0, 0, 11, 13, 16, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 10, 33, 4], [35, 1, 0, 3, 0, 0, 10, 0, 0, 0, 0, 36, 0, 0, 0, 0, 0, 39, 0, 0, 2, 0, 1, 35], [34, 10, 0, 3, 0, 0, 38, 0, 0, 0, 0, 0, 26, 4, 1, 0, 8, 0, 33, 0, 0, 0, 0, 0], [0, 24, 0, 18, 0, 0, 10, 31, 0, 0, 0, 0, 0, 29, 0, 0, 0, 6, 0, 36, 0, 0, 0, 0], [0, 2, 31, 8, 1, 0, 0, 6, 35, 0, 5, 0, 2, 2, 21, 6, 0, 8, 0, 0, 43, 0, 0, 0], [2, 1, 3, 25, 3, 1, 0, 0, 1, 26, 6, 0, 0, 1, 4, 34, 6, 1, 0, 0, 0, 31, 4, 0], [4, 4, 3, 6, 23, 1, 0, 0, 4, 2, 34, 0, 0, 2, 0, 3, 33, 5, 0, 0, 0, 1, 41, 0], [8, 0, 0, 11, 0, 14, 4, 1, 1, 0, 7, 29, 0, 0, 5, 1, 4, 28, 0, 0, 0, 0, 0, 51]]
acc_matrix = [[0.9416666626930237, 0.8166666626930237, 0.512499988079071, 0.5458333492279053], [0.8291666507720947, 0.9750000238418579, 0.4333333373069763, 0.7875000238418579], [0.3125, 0.6166666746139526, 1.0, 0.7166666388511658], [0.6291666626930237, 0.8041666746139526, 0.7124999761581421, 0.9791666865348816]]
print('Fourier Treatment')

print('ACCURACY MATRIX')
for row in acc_matrix:
    for value in row:
        print(f"{value:2f}", end=" ")  # Adjust the formatting as needed
    print()  # Move to the next row

print('CONFUSION MATRIX')
for row in conf_matrix:
    for value in row:
        print(f"{value:2d}", end=" ")  # Adjust the formatting as needed
    print()  # Move to the next row




# All accelerations and rot_freq considered (Fourier Treatment)
conf_matrix = [[37, 0, 1, 0, 0, 0, 17, 18, 0, 0, 0, 0, 9, 8, 4, 0, 2, 13, 16, 6, 18, 0, 0, 0], [1, 26, 1, 0, 10, 0, 8, 28, 0, 0, 0, 0, 4, 5, 14, 1, 4, 3, 0, 14, 19, 1, 0, 0], [3, 0, 35, 1, 0, 1, 0, 0, 35, 0, 11, 1, 0, 3, 24, 0, 0, 9, 0, 1, 30, 1, 1, 3], [0, 0, 0, 41, 0, 0, 0, 5, 10, 5, 13, 10, 1, 0, 25, 3, 2, 13, 0, 0, 18, 6, 9, 9], [0, 0, 1, 2, 32, 0, 0, 3, 4, 7, 24, 1, 1, 0, 25, 0, 4, 13, 2, 1, 19, 3, 4, 10], [0, 0, 0, 0, 1, 47, 1, 0, 6, 0, 8, 25, 0, 0, 31, 2, 0, 17, 0, 0, 1, 12, 2, 34], [11, 12, 0, 0, 15, 5, 31, 1, 0, 0, 0, 0, 0, 6, 33, 0, 1, 0, 21, 9, 4, 0, 6, 2], [6, 22, 1, 1, 1, 0, 0, 42, 0, 0, 0, 0, 0, 7, 37, 0, 0, 0, 10, 14, 10, 1, 9, 1], [0, 1, 31, 2, 0, 8, 0, 1, 45, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 24, 1, 1, 19], [0, 0, 14, 11, 14, 3, 0, 2, 0, 34, 0, 0, 0, 0, 34, 1, 1, 0, 0, 0, 13, 10, 5, 11], [0, 1, 7, 6, 11, 15, 0, 0, 0, 2, 39, 0, 3, 1, 29, 0, 7, 0, 0, 0, 10, 4, 12, 10], [0, 1, 8, 3, 0, 30, 0, 0, 0, 0, 0, 43, 0, 0, 40, 0, 0, 12, 0, 0, 5, 3, 0, 25], [16, 7, 5, 3, 1, 0, 0, 9, 1, 24, 2, 0, 38, 0, 0, 0, 0, 0, 26, 6, 3, 0, 4, 0], [2, 13, 14, 5, 6, 0, 0, 7, 0, 28, 0, 0, 0, 41, 0, 0, 0, 0, 1, 33, 0, 1, 0, 0], [0, 4, 18, 11, 10, 2, 0, 7, 5, 33, 0, 0, 0, 0, 44, 0, 1, 0, 1, 5, 38, 0, 3, 0], [0, 5, 3, 15, 6, 0, 0, 0, 0, 39, 0, 0, 0, 0, 0, 39, 3, 0, 0, 7, 7, 19, 4, 1], [3, 1, 11, 3, 23, 1, 0, 0, 0, 38, 0, 0, 0, 0, 1, 1, 28, 1, 0, 13, 11, 8, 12, 0], [0, 9, 5, 23, 0, 15, 0, 0, 0, 47, 0, 0, 0, 0, 0, 1, 1, 41, 3, 7, 14, 4, 0, 9], [32, 6, 0, 4, 0, 0, 18, 19, 0, 0, 3, 0, 8, 15, 17, 0, 1, 2, 42, 0, 0, 0, 0, 0], [19, 18, 0, 0, 0, 0, 9, 17, 0, 3, 3, 0, 0, 28, 6, 0, 0, 9, 1, 42, 0, 1, 0, 0], [0, 11, 16, 0, 17, 0, 0, 0, 21, 7, 1, 10, 3, 6, 32, 0, 0, 0, 0, 0, 32, 1, 0, 11], [0, 3, 7, 8, 21, 4, 0, 0, 0, 32, 7, 8, 8, 1, 15, 7, 2, 1, 0, 0, 0, 38, 0, 0], [0, 0, 16, 2, 21, 3, 0, 3, 3, 4, 24, 0, 13, 1, 12, 1, 12, 0, 0, 0, 1, 1, 38, 0], [2, 12, 13, 2, 3, 0, 0, 0, 1, 0, 0, 47, 0, 0, 9, 3, 0, 28, 0, 0, 0, 0, 0, 32]]
acc_matrix = [[0.9083333611488342, 0.5583333373069763, 0.25833332538604736, 0.4333333373069763], [0.4833333194255829, 0.9750000238418579, 0.2291666716337204, 0.4416666626930237], [0.4166666567325592, 0.21250000596046448, 0.9624999761581421, 0.5708333253860474], [0.3958333432674408, 0.6625000238418579, 0.4791666567325592, 0.9333333373069763]]
print('Raw Data (no treatment)')

print('ACCURACY MATRIX')
for row in acc_matrix:
    for value in row:
        print(f"{value:2f}", end=" ")  # Adjust the formatting as needed
    print()  # Move to the next row

print('CONFUSION MATRIX')
for row in conf_matrix:
    for value in row:
        print(f"{value:2d}", end=" ")  # Adjust the formatting as needed
    print()  # Move to the next row
