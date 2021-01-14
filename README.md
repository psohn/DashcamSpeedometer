# DashcamSpeedometer
A CNN trained to predict vehicle speed based on dashcam footage from comma.ai's [dashcam speed challenge](https://github.com/commaai/speedchallenge) using dense optical flow calculations. The dashcam footage was in 20 fps and had a recorded speed for each frame. 

## Preprocessing Video

Look at the base image for objects that might be relevant.

![base image](https://i.gyazo.com/e0c9b53560740da36e707db4262dbc15.png)

Use the road lines as markers by filtering for road line colours.

![white yellow filter](https://i.gyazo.com/2647e9b00aef7c82835cda6b0e5c1212.png)

Find canny edges on the road lines.

![canny image](https://i.gyazo.com/4701b0438625ace6b8498c101523d939.png)

Apply a mask on the road to filter out the sky and dashboard.

![mask image](https://i.gyazo.com/91733a0c52680d96cbf2f692c47ff09d.png)

Reapply the canny edges but to the original image.

![preprocessed image](https://i.gyazo.com/0a982486f6cb198f28108972d24d9848.png)

Calculate the optical flow of consecutive frames.

![optical flow image](https://i.gyazo.com/e1cf7804b76caed705c36a05d485c363.png)

![masked optical flow image](https://i.gyazo.com/ee496d19fcd0123b2eeacb317b2cd98d.png)

## Training

A standard CNN model was employed using the Nvidia system taken from [another repository](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector). The training whilst holding the video in memory was too much for my computer so code had to be used to delete variables as the model was trained.

```python
# clear RAM
del video_preprocess
```

## Evaluation

Using a 30% validation set, an MSE of 7.7 was achieved after 8 epochs of training.
