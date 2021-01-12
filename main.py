# import libraries
from model_utils import *
from optical_flow_utils import *

# preprocess video with optical flow
video_preprocess = preprocess_video('train')

# import speed values
speed_0 = np.loadtxt('data/train.txt')[0]
speeds = np.loadtxt('data/train.txt')[1:]


''' uncomment for training '''
# # train test split
# X_train, X_test, y_train, y_test = train_test_split(video_preprocess, speeds, test_size = 0.3, random_state = 42)

# # clear RAM
# del video_preprocess

# # start training
# initiate_model(X_train, y_train, 10, validation = (X_test, y_test))

''' uncomment for prediction '''
# prediction
model = load_model('model/model')
pred = model.predict(video_preprocess)
pred = pred.reshape((pred.shape[0]))
pred_0 = np.array([pred[0]])
pred = np.append(pred_0, pred)
pred = pd.DataFrame(pred)
pred.to_csv('data/train_pred.csv')