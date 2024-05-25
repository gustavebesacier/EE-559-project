import matplotlib.pyplot as plt

# Data
epochs = list(range(11))

#jews
train_accuracy = [
    0.5753472222222222, 0.525, 0.5, 0.4986111111111111, 0.5034722222222222,
    0.4951388888888889, 0.49583333333333335, 0.49166666666666664,
    0.5090277777777777, 0.4951388888888889, 0.4951388888888889
]
train_f1 = [
    0.5676917638741604, 0.575682382133995, 0.6106003244997296, 0.6112008616047389,
    0.5806451612903226, 0.6623316302833256, 0.6090468497576736, 0.5746658919233004,
    0.6039215686274509, 0.6623316302833256, 0.6623316302833256
]
test_accuracy = [
    0.7222222222222222, 0.5194444444444445, 0.5194444444444445, 0.48055555555555557,
    0.5194444444444445, 0.5194444444444445, 0.5194444444444445, 0.5194444444444445,
    0.5194444444444445, 0.5194444444444445, 0.5194444444444445
]
test_f1 = [
    0.6575342465753424, 0.6837294332723949, 0.6837294332723949, 0.0,
    0.6837294332723949, 0.6837294332723949, 0.6837294332723949, 0.6837294332723949,
    0.6837294332723949, 0.6837294332723949, 0.6837294332723949
]

#asian
train_accuracy = [
    0.4951171875, 0.4931640625, 0.4775390625, 0.4912109375, 0.4833984375,
    0.5068359375, 0.4970703125, 0.4716796875, 0.5029296875, 0.4912109375,
    0.4892578125
]
train_f1 = [
    0.5436893203883495, 0.5378450578806767, 0.4253490870032223, 0.5669160432252701,
    0.3138780804150454, 0.4965104685942173, 0.5347786811201445, 0.4648862512363996,
    0.6692657569850552, 0.6132145508537491, 0.6293408929836996
]
test_accuracy = [
    0.51171875, 0.51171875, 0.51171875, 0.51171875, 0.48828125,
    0.51171875, 0.51171875, 0.48828125, 0.48828125, 0.48828125,
    0.48828125
]
test_f1 = [
    0.0, 0.0, 0.0, 0.0, 0.6561679790026247,
    0.0, 0.0, 0.6561679790026247, 0.6561679790026247, 0.6561679790026247,
    0.6561679790026247
]

#MUSLIM
train_accuracy = [
    0.5223214285714286, 0.49970238095238095, 0.5086309523809524, 0.5092261904761904,
    0.5038690476190476, 0.4994047619047619, 0.4979166666666667, 0.49970238095238095,
    0.5038690476190476, 0.5038690476190476, 0.5038690476190476
]
train_f1 = [
    0.5682001614205004, 0.5811113879890356, 0.6299036090562654, 0.6355801104972376,
    0.6700969720957847, 0.6446979298690325, 0.6092193652999769, 0.6387277025574898,
    0.6700969720957847, 0.6700969720957847, 0.6637078878353843
]
test_accuracy = [
    0.5154761904761904, 0.4845238095238095, 0.4845238095238095, 0.4845238095238095,
    0.4845238095238095, 0.5154761904761904, 0.4845238095238095, 0.4845238095238095,
    0.4845238095238095, 0.4845238095238095, 0.4845238095238095
]
test_f1 = [
    0.0, 0.652766639935846, 0.652766639935846, 0.652766639935846,
    0.652766639935846, 0.0, 0.652766639935846, 0.652766639935846,
    0.652766639935846, 0.652766639935846, 0.652766639935846
]

#Disabilities
train_accuracy = [
    0.4931640625, 0.4951171875, 0.4912109375, 0.5029296875, 0.5029296875,
    0.5087890625, 0.5087890625, 0.5029296875, 0.5029296875, 0.4931640625,
    0.5126953125
]
train_f1 = [
    0.5605419136325148, 0.49659201557935734, 0.6429061000685401, 0.5768911055694098,
    0.6692657569850552, 0.6589830508474577, 0.5927125506072874, 0.6692657569850552,
    0.6692657569850552, 0.6004618937644342, 0.5738684884713919
]
test_accuracy = [
    0.48828125, 0.48828125, 0.48828125, 0.48828125, 0.48828125,
    0.51171875, 0.48828125, 0.48828125, 0.48828125, 0.51171875,
    0.51171875
]
test_f1 = [
    0.6561679790026247, 0.6561679790026247, 0.6561679790026247, 0.6561679790026247,
    0.6561679790026247, 0.0, 0.6561679790026247, 0.6561679790026247,
    0.6561679790026247, 0.0, 0.0
]
"""
# Plotting
plt.figure(figsize=(12, 6))

plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, train_f1, label='Training F1', marker='o')
plt.plot(epochs, test_accuracy, label='Test Accuracy', marker='o')
plt.plot(epochs, test_f1, label='Test F1', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training and Test Performance over Epochs')
plt.legend()
plt.grid(True)
plt.show()
"""
epochs = list(range(1, 8))
#jew
train_accuracy = [
    0.527734375, 0.748046875, 0.803515625, 0.85078125, 0.890234375, 0.91171875, 0.924609375
]
train_f1 = [
    0.6447252424331472, 0.7191989551589029, 0.8088179399467883, 0.8591445427728613,
    0.8966531813166605, 0.916972814107274, 0.9287559985234404
]

test_accuracy = [
    0.7390625, 0.7453125, 0.75625, 0.7328125, 0.740625, 0.7375, 0.7390625
]
test_f1 = [
    0.6639839034205232, 0.7400318979266348, 0.75625, 0.7532467532467533, 0.75,
    0.7507418397626113, 0.7547723935389133
]

train_losses = [
    0.6959974359720945, 0.5439712979830802, 0.4838130371528678, 0.4195637660450302,
    0.33783771423622966, 0.30053555908380075, 0.27127442422206516
]
distillation_losses = [
    -0.28796178139746187, -0.5076098173856736, -0.6274181945249439, -0.7351680541411042,
    -0.8171452697366476, -0.855062466673553, -0.8756826370954514
]
test_losses = [
    0.5361276946961879, 0.5297100558876991, 0.5806654628366232, 0.6385499568656087,
    0.7565880540758372, 0.8194123767316341, 0.8574221080169082
]
#women
#train_accuracy = [0.5875, 0.7608630952380953, 0.8068452380952381, 0.8436011904761904, 0.8741071428571429, 0.8967261904761905, 0.9130952380952381]
#train_f1 = [0.6500883615248675, 0.7707887605191841, 0.8159387407827566, 0.8510698597137594, 0.8791083166619034, 0.9005160550458715, 0.915850144092219]

#test_accuracy = [0.7309523809523809, 0.7464285714285714, 0.7357142857142858, 0.7380952380952381, 0.7386904761904762, 0.7321428571428571, 0.7351190476190477]
#test_f1 = [0.7457817772778402, 0.772192513368984, 0.748868778280543, 0.7649572649572649, 0.7475560667050029, 0.7543668122270742, 0.7523650528658876]

#train_losses = [0.6767292723059655, 0.5709068461188248, 0.5014120474546438, 0.42714473893865945, 0.3637928392405489, 0.31442632934832504, 0.26984222999862617]
#distillation_losses = [-0.3386127735177676, -0.5480139855827604, -0.6450277843645641, -0.7280677547057469, -0.7904963124366033, -0.8342158900130363, -0.8656014761044866]
#test_losses = [0.5691186598369053, 0.606432727546919, 0.6831342039363725, 0.7221764658888181, 0.7958105098633539, 0.8886024047931035, 0.9370769198451724]


# Plotting accuracy and F1 score  ### PLOT 1 ####
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, train_f1, label='Training F1', marker='o')
plt.plot(epochs, test_accuracy, label='Test Accuracy', marker='o')
plt.plot(epochs, test_f1, label='Test F1', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training and Test Accuracy/F1 over Epochs')
plt.legend()
plt.grid(True)

# Plotting losses
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.plot(epochs, distillation_losses, label='Distillation Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Distillation, and Test Loss over Epochs')
plt.legend()
plt.grid(True)

plt.savefig('jews_metrics.png', dpi=300)
plt.tight_layout()
plt.show()

# plot temperature perf ###  PLOT 2 ###
# 10 T
temperatures = [0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 0.1]
train_f1_scores =[0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 0.1] # replace
test_f1_scores =[0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 0.1]  # replace
train_accuracies =[0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 0.1] # replace
test_accuracies =[0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 0.1] # replace

# Plotting
plt.figure(figsize=(10, 6))

# Plot F1 Scores
plt.plot(temperatures, train_f1_scores, marker='o', label='Train F1 Score', linestyle='-')
plt.plot(temperatures, test_f1_scores, marker='o', label='Test F1 Score', linestyle='--')

# Plot Accuracies
plt.plot(temperatures, train_accuracies, marker='s', label='Train Accuracy', linestyle='-')
plt.plot(temperatures, test_accuracies, marker='s', label='Test Accuracy', linestyle='--')

# Labels and Title
plt.xlabel('Temperature')
plt.ylabel('Score')
plt.title('F1 Score and Accuracy vs Temperature')
plt.legend()
plt.grid(True)

# save plot
plt.savefig('muslim_T.png', dpi=300)
plt.tight_layout()
plt.show()







