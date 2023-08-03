
#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import os
import glob
import cv2
import numpy as np
from PIL import Image
from boxmot import DeepOCSORT
from pathlib import Path
from yolo import YOLO

import torch
import torchvision
from torch.utils.data import  Dataset,DataLoader
from  torchvision.transforms import transforms

# os.environ["CUDA_VISIBLE_DEVICES"] = '10'


if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #mode = "predict"
    #mode = "quantize"
    #mode = "detect_video"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    #crop            = False
    #count           = False
    crop            = True
    count           = True
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 'v4cut100_150.mp4'
    video_save_path = "v4cut100_150_output.mp4"
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/000007.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "quantize":
        image = Image.open('000015.jpg')
        quant_mode="calib"
        #quant_mode="test"
        quantize_dir="./quantized_model/"
        
        shell_cmd="mkdir ./quantized_model"

        if (os.path.exists(quantize_dir))==False:
            print("The dir:quantized_model is not exit,need create!!\n")
            if os.system(shell_cmd) == 0:
                print("create quantized_model success")
            else:
                print("create quantized_model falid")

        yolo.quantize(image,quantize_dir,quant_mode,count=count)

    elif mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
       
        # img = input('Input image filename:')
        image = Image.open('000015.jpg')
        # 可以得到预测截取的图像
        r_image = yolo.detect_image(image,crop=crop,count=count)
        #print(r_image)
        r_image.save("output.jpg")
        #r_image.show()

        print("Predict Detection Done!")

    elif mode == "video":
       
       tracker = DeepOCSORT(model_weights=Path('mobilenetv2_x1_4_dukemtmcreid.pt'),  # which ReID model to use
                        device='cuda:1',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
                        fp16=True,  # wether to run the ReID model with half precision or not
                        )
       
       capture = cv2.VideoCapture(video_path)
       
       if video_save_path!="":
           fourcc  = cv2.VideoWriter_fourcc(*'XVID')
           size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
           out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

       ref, frame = capture.read()
       if not ref:
           raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

       fps = 0.0
       color = (0, 0, 255)  # BGR
       thickness = 2
       fontscale = 0.5
       count = 0
       while(True):
           # 时间起始
           t1 = time.time()
           # 读取图片
           ref, im_cv2 = capture.read()
           if not ref:
               break
           # 转换为Image
           im = Image.fromarray(np.uint8(im_cv2))
           
           detections = yolo.track(im) # detections存入xmin,ymin,xmax,ymax ,conf ,cls
           print(detections)
           # 若没有检测到目标(待优化,遮挡等)
           if isinstance(detections,Image.Image) :
               # ts = tracker.update([],im_cv2)
               print('no detections!')
           else : 
               # boat类别我直接取定值0(待优化)
               # dets = np.array([[ymax, xmin, ymin, xmax,conf,0]])
               # detections 内容为xmin,ymin,xmax,ymax,conf,cls
                dets = []
                # ---------------------- #
                detections_np = np.array(detections) # 本来是这步可以直接输入tracker.update(),但是种类这边会出问题。先用下面的处理代替
                # ---------------------- #
                for detection in detections:
                    det = [[detection[0],detection[1],detection[2],detection[3],detection[4],0]] # detection[5]是种类,未加入(待优化,在ocsort.py里,种类在kalman滤波器里还要用到)
                    print('det:',det)
                    det_np = np.array(det)
                
                    # 更新tracker
                    ts = tracker.update(det_np, im_cv2)
                    print(count,"Tracker更新完毕!")
                    # 取跟踪更新值
                    xyxys = ts[:, 0:4].astype('int') # float64 to int
                    # print('xyxys:',xyxys)
                    ids = ts[:, 4].astype('int') # float64 to int
                    confs = ts[:, 5]
                    clss = ts[:, 6]

                    # 画画
                    if ts.shape[0] != 0:
                        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                            im_cv2 = cv2.rectangle(
                            im_cv2,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            color,  
                            thickness
                    )
                        cv2.putText(
                        im_cv2,
                        f'id: {id}, conf: {conf}, c: {cls}',
                        (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        color,
                        thickness
                    )

                    if video_save_path!="":
                        out.write(im_cv2)
                        print('真的有画进去!')
                        count +=1

           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       # else : # 如果未检测到目标 (待优化)
           

       print("Video Detection Done!")
       capture.release()
       if video_save_path!="":
           print("Save processed video to the path :" + video_save_path)
           out.release()
       cv2.destroyAllWindows()
       
    elif mode == "detect_video":
        capture = cv2.VideoCapture(video_path)
        
        ref, frame = capture.read()
        if not ref:
           raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        color = (0, 0, 255)  # BGR
        thickness = 2
        fontscale = 0.5
        count = 0
        while(True):
            # 时间起始
            t1 = time.time()
            # 读取图片
            ref, im_cv2 = capture.read()
            if not ref:
                break
            # 转换为Image
            im = Image.fromarray(np.uint8(im_cv2))
           
            detections = yolo.track(im) # detections存入xmin,ymin,xmax,ymax ,conf ,cls
           

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
