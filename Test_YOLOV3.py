from face_YOLO_model3 import *
from face_losses import *
from face_draw_samples import *
import cv2
import numpy as np   
import time
import argparse

yolo_anchors = np.array([(0.14516129, 0.188), (0.24782608, 0.45401174), (0.23333333, 0.23641565),
                         (0.675, 0.74827874), (0.37395832, 0.6447896), (0.30808082, 0.2969374),
                         (0.38565022, 0.382), (0.470389, 0.468), (0.5615616, 0.576)], dtype=np.float32)
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
optim = tf.keras.optimizers.Adam(0.001)

def test_(frames):

    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    frames = tf.cast(frames, tf.float32) / 255.
    images = tf.expand_dims(frames, 0)
    h = model(images, False)
    output_0 = h[0]
    output_1 = h[1]
    output_2 = h[2]

    boxes_0 = Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[0]], 40),
                        name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[1]], 40),
                        name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[2]], 40),
                        name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, yolo_anchors, yolo_anchor_masks, 40, 100),
                        name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    #
    boxes, scores, classes, nums = outputs
    #

    return boxes, scores, classes, nums

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Expression real-time test with Blazeface detection')
    parser.add_argument('--device', default='cpu', choices = ['cpu', 'cuda'])
    parser.add_argument('--prototype', type=int, default = 1, choices=[1,2,3])
    opt = parser.parse_args()

    ## 모델 만들기
    #model = ResLSTMNet(BasicBlock1, [1, 2, 5, 3])

    model = YoloV3(416, 3, masks=yolo_anchor_masks, classes=40)

    model_pretrained = YoloV3(416, 3, masks=yolo_anchor_masks, classes=40)
    model_pretrained.load_weights("C:/Users/Yuhwan/Downloads/ck/yolov3.tf")
    model.get_layer('yolo_darknet').set_weights(
        model_pretrained.get_layer('yolo_darknet').get_weights())
    freeze_all(model.get_layer('yolo_darknet'))
    x_36, x_61, x = model.output

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(yolo_anchor_masks[0]), 40, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(yolo_anchor_masks[1]), 40, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(yolo_anchor_masks[2]), 40, name='yolo_output_2')(x)

    model = tf.keras.Model(model.input, (output_0, output_1, output_2), name='yolov3')
    model.summary()

    ckpt = tf.train.Checkpoint(model=model, optim=optim)
    ckpt_manager = tf.train.CheckpointManager(ckpt, "C:/Users/Yuhwan/Downloads/12", 5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored!!!!!!!")

    # Video capture using webcam
    camera = cv2.VideoCapture(0)
    
    iter_Num = 0
    d_time = 0.0
    p_time = 0.0

    flags = True
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    while True:
        s4 = time.time()
        ymins = []
        xmins = []
        ymaxs = []
        xmaxs = []

        while cv2.waitKey(200) < 0:
            # Capture image from camera
            ret, frame = camera.read()  # ret: frame 사용 가능 여부
                                        # frame: 캡쳐된 image arr vector

            frame = cv2.resize(frame, (416, 416))
            if flags:
                video = cv2.VideoWriter("C:/Users/Yuhwan/Downloads/test.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                flags = False
            


            # to gray scale
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, objectness, classes, nums = test_(frame)

            #img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
            wh = np.flip(frame.shape[0:2])
            if nums != 0:
                for i in range(nums):
                    x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
                    x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
                    img = cv2.rectangle(frame, x1y1, x2y2, (0, 0, 255), 2)
                    cv2.imshow('Cam', img)
                    video.write(frame)

            #d_time += time.time()-s4
            #iter_Num += 1
            #print("[*]iter ",iter_Num)
            #print('Avg detection time: {:.3f} sec'.format(d_time/iter_Num))
        
            #preds_sum = np.zeros(3)
            #if xmin !=0 and ymin !=0 and xmax !=0 and ymax !=0:
            #    crop_img = gray[ymin:ymax, xmin:xmax]
            #    crop_img = cv2.resize(crop_img, (64, 64), crop_img)
            #    frame_ = np.expand_dims(crop_img, 0)
            #    frame_ = np.expand_dims(frame_, 3)
            #    frame_ = np.transpose(frame_, (0, 3, 1, 2))
            #    preds, label = predect_emotion(opt, model,frame_)
            #    preds_sum += preds        
            #    # Assign labeling
            #    cv2.putText(frame, EMOTIONS[label], (xmax, ymax - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            #    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            #    # Create empty image
            #    canvas = np.zeros((250, 300, 3), dtype="uint8")
            #    # Label printing
            #    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds_sum/6)):
            #        text = "{}: {:.2f}%".format(emotion, prob * 100)    
            #        w_ = int(prob * 300)
            #        cv2.rectangle(canvas, (7, (i * 35) + 5), (w_, (i * 35) + 35), (255, 0, 0), -1)
            #        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Open two windows
        ## Display image ("Emotion Recognition")
        ## Display probabilities of emotion
            else:
                cv2.imshow('Cam', frame)
                video.write(frame)

        # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clear program and close windows
    camera.release()
    cv2.destroyAllWindows()
