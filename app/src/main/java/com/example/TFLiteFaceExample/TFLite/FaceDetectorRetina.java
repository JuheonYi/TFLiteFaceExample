package com.example.TFLiteFaceExample.TFLite;

import android.app.Activity;
import android.os.SystemClock;
import android.util.Log;

import com.example.TFLiteFaceExample.boundingBox;

import java.io.IOException;

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class FaceDetectorRetina extends FaceDetector {

    public FaceDetectorRetina(Activity activity, int h, int w, int c, float s, String model_name) throws IOException {
        super(activity, h, w, c, model_name);
        Log.d("johnyi-TFLite", String.format("Creating RetinaFaceDetector, input: %dx%d",h,w));

        scale = s;

        h_stride32 = (int)Math.ceil((double)h/32);
        w_stride32 = (int)Math.ceil((double)w/32);

        h_stride16 = (int)Math.ceil((double)h/16);
        w_stride16 = (int)Math.ceil((double)w/16);

        h_stride8 = (int)Math.ceil((double)h/8);
        w_stride8 = (int)Math.ceil((double)w/8);

        this.out_h = h/32;
        this.out_w = w/32;

        score_stride8 = new float[1][h_stride8][w_stride8][4];
        bbox_deltas_stride8 = new float[1][h_stride8][w_stride8][8];
        landmark_deltas_stride8 = new float[1][h_stride8][w_stride8][20];

        score_stride16 = new float[1][h_stride16][w_stride16][4];
        bbox_deltas_stride16 = new float[1][h_stride16][w_stride16][8];
        landmark_deltas_stride16 = new float[1][h_stride16][w_stride16][20];

        score_stride32 = new float[1][h_stride32][w_stride32][4];
        bbox_deltas_stride32 = new float[1][h_stride32][w_stride32][8];
        landmark_deltas_stride32 = new float[1][h_stride32][w_stride32][20];
    }

    @Override
    protected String getModelPath() {
        if(use_padding) return String.format("RetinaFace%s-%dx%d.tflite",model_name,h+2*p,w+2*p);
        else return String.format("RetinaFace%s-%dx%d.tflite",model_name,h,w);
    }

    float scale;

    int num_anchors_stride32 = 2;
    int h_stride32;
    int w_stride32;
    float[][] anchors_fpn_stride32 = {{-248.f, -248.f, 263.f, 263.f}, {-120.f, -120.f,  135.f,  135.f}};

    int num_anchors_stride16 = 2;
    int h_stride16;
    int w_stride16;
    float[][] anchors_fpn_stride16 = {{-56.f, -56.f,  71.f,  71.f}, {-24.f, -24.f,  39.f,  39.f}};

    int num_anchors_stride8 = 2;
    int h_stride8;
    int w_stride8;
    float[][] anchors_fpn_stride8 = {{-8.f, -8.f, 23.f, 23.f}, {0.f,  0.f, 15.f, 15.f}};

    float prob_thresh = 0.2f;

    float[][][][] score_stride8, bbox_deltas_stride8, landmark_deltas_stride8;
    float[][][][] score_stride16, bbox_deltas_stride16, landmark_deltas_stride16;
    float[][][][] score_stride32, bbox_deltas_stride32, landmark_deltas_stride32;

    //need anchors_fpn -> anchors
    @Override
    public void calc_bboxes(){
        float x1, y1, x2, y2;
        float width, height, ctr_x, ctr_y;
        float pred_w, pred_h, pred_ctr_x, pred_ctr_y;
        float score1, score2;
        float score;
        int stride;
        float[] landmarks = new float[10];

        ///*
        //stride 32
        stride = 32;
        for(int i = 0; i < h_stride32; i++){
            for(int j = 0; j < w_stride32; j++){
                for(int k = 0; k < num_anchors_stride32; k++) {
                    score1 = score_stride32[0][i][j][k];
                    score2 = score_stride32[0][i][j][num_anchors_stride32+k];
                    score = (float)Math.exp((double)score2)/((float)Math.exp((double)score1)+(float)Math.exp((double)score2));
                    if (score >= prob_thresh){
                        width = anchors_fpn_stride32[k][2] - anchors_fpn_stride32[k][0] + 1;
                        height = anchors_fpn_stride32[k][3] - anchors_fpn_stride32[k][1] + 1;
                        ctr_x = j*stride + anchors_fpn_stride32[k][0] + 0.5f*(width-1);
                        ctr_y = i*stride + anchors_fpn_stride32[k][1] + 0.5f*(height-1);

                        pred_ctr_x = bbox_deltas_stride32[0][i][j][4*k] * width + ctr_x;
                        pred_ctr_y = bbox_deltas_stride32[0][i][j][4*k+1] * height + ctr_y;
                        pred_w = (float)Math.exp((double)bbox_deltas_stride32[0][i][j][4*k+2]) * width;
                        pred_h = (float)Math.exp((double)bbox_deltas_stride32[0][i][j][4*k+3]) * height;

                        x1 = pred_ctr_x - 0.5f * (pred_w - 1);
                        y1 = pred_ctr_y - 0.5f * (pred_h - 1);
                        x2 = pred_ctr_x + 0.5f * (pred_w - 1);
                        y2 = pred_ctr_y + 0.5f * (pred_h - 1);

                        //calc landmarks
                        for(int l = 0; l < 5; l++){
                            landmarks[2*l] = landmark_deltas_stride32[0][i][j][10*k+2*l]*width + ctr_x;
                            landmarks[2*l+1] = landmark_deltas_stride32[0][i][j][10*k+2*l+1]*height + ctr_y;
                        }
                        bboxes.add(bboxes.size(), new boundingBox(scale*x1, scale*y1, scale*x2, scale*y2, score, landmarks));
                    }
                }
            }
        }

        //stride 16
        stride = 16;
        for(int i = 0; i < h_stride16; i++){
            for(int j = 0; j < w_stride16; j++){
                for(int k = 0; k < num_anchors_stride16; k++) {
                    score1 = score_stride16[0][i][j][k];
                    score2 = score_stride16[0][i][j][num_anchors_stride32+k];
                    score = (float)Math.exp((double)score2)/((float)Math.exp((double)score1)+(float)Math.exp((double)score2));
                    if (score >= prob_thresh){
                        width = anchors_fpn_stride16[k][2] - anchors_fpn_stride16[k][0] + 1;
                        height = anchors_fpn_stride16[k][3] - anchors_fpn_stride16[k][1] + 1;
                        ctr_x = j*stride + anchors_fpn_stride16[k][0] + 0.5f*(width-1);
                        ctr_y = i*stride + anchors_fpn_stride16[k][1] + 0.5f*(height-1);

                        pred_ctr_x = bbox_deltas_stride16[0][i][j][4*k] * width + ctr_x;
                        pred_ctr_y = bbox_deltas_stride16[0][i][j][4*k+1] * height + ctr_y;
                        pred_w = (float)Math.exp((double)bbox_deltas_stride16[0][i][j][4*k+2]) * width;
                        pred_h = (float)Math.exp((double)bbox_deltas_stride16[0][i][j][4*k+3]) * height;

                        x1 = pred_ctr_x - 0.5f * (pred_w - 1);
                        y1 = pred_ctr_y - 0.5f * (pred_h - 1);
                        x2 = pred_ctr_x + 0.5f * (pred_w - 1);
                        y2 = pred_ctr_y + 0.5f * (pred_h - 1);

                        //calc landmarks
                        for(int l = 0; l < 5; l++){
                            landmarks[2*l] = landmark_deltas_stride16[0][i][j][10*k+2*l]*width + ctr_x;
                            landmarks[2*l+1] = landmark_deltas_stride16[0][i][j][10*k+2*l+1]*height + ctr_y;
                        }
                        bboxes.add(bboxes.size(), new boundingBox(scale*x1, scale*y1, scale*x2, scale*y2, score, landmarks));
                    }
                }
            }
        }
        //*/

        //stride 8
        stride = 8;
        for(int i = 0; i < h_stride8; i++){
            for(int j = 0; j < w_stride8; j++){
                for(int k = 0; k < num_anchors_stride8; k++) {
                    score1 = score_stride8[0][i][j][k];
                    score2 = score_stride8[0][i][j][num_anchors_stride32+k];
                    score = (float)Math.exp((double)score2)/((float)Math.exp((double)score1)+(float)Math.exp((double)score2));
                    if (score >= prob_thresh){
                        width = anchors_fpn_stride8[k][2] - anchors_fpn_stride8[k][0] + 1;
                        height = anchors_fpn_stride8[k][3] - anchors_fpn_stride8[k][1] + 1;
                        ctr_x = j*stride + anchors_fpn_stride8[k][0] + 0.5f*(width-1);
                        ctr_y = i*stride + anchors_fpn_stride8[k][1] + 0.5f*(height-1);

                        pred_ctr_x = bbox_deltas_stride8[0][i][j][4*k] * width + ctr_x;
                        pred_ctr_y = bbox_deltas_stride8[0][i][j][4*k+1] * height + ctr_y;
                        pred_w = (float)Math.exp((double)bbox_deltas_stride8[0][i][j][4*k+2]) * width;
                        pred_h = (float)Math.exp((double)bbox_deltas_stride8[0][i][j][4*k+3]) * height;

                        x1 = pred_ctr_x - 0.5f * (pred_w - 1);
                        y1 = pred_ctr_y - 0.5f * (pred_h - 1);
                        x2 = pred_ctr_x + 0.5f * (pred_w - 1);
                        y2 = pred_ctr_y + 0.5f * (pred_h - 1);

                        //calc landmarks
                        for(int l = 0; l < 5; l++){
                            landmarks[2*l] = landmark_deltas_stride8[0][i][j][10*k+2*l]*width + ctr_x;
                            landmarks[2*l+1] = landmark_deltas_stride8[0][i][j][10*k+2*l+1]*height + ctr_y;
                        }
                        //if(x1 >= 0 && x2 < w && y1 >= 0 && y2 < h) {
                        bboxes.add(bboxes.size(), new boundingBox(scale*x1, scale*y1, scale*x2, scale*y2, score, landmarks));
                        //}
                    }
                }
            }
        }

    }

    @Override
    public Vector<boundingBox> detect(byte[] img_in, long[] times){
        imgData.rewind();

        startTime = SystemClock.uptimeMillis();
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                for(int k = 0; k < c; k++){
                    imgData.putFloat((float)(img_in[i*w*c + j*c + k]&0xff));
                }
            }
        }

        endTime = SystemClock.uptimeMillis();
        times[0] = endTime-startTime;

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();

        outputMap.put(0, score_stride32);
        outputMap.put(1, bbox_deltas_stride32);
        outputMap.put(2, landmark_deltas_stride32);
        outputMap.put(3, score_stride16);
        outputMap.put(4, bbox_deltas_stride16);
        outputMap.put(5, landmark_deltas_stride16);
        outputMap.put(6, score_stride8);
        outputMap.put(7, bbox_deltas_stride8);
        outputMap.put(8, landmark_deltas_stride8);

        startTime = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputArray, outputMap);
        endTime = SystemClock.uptimeMillis();
        times[1] = endTime-startTime;
        //Log.d("johnyi-TFLite", "inference done!, "+ Long.toString(endTime - startTime));
        startTime = SystemClock.uptimeMillis();
        //Log.d("johnyi-TFLite","calc bboxes");
        calc_bboxes();
        //Log.d("johnyi-TFLite","NMS");
        nonMaximumSuppression();
        endTime = SystemClock.uptimeMillis();
        times[2] = endTime-startTime;
        //Log.d("johnyi-TFLite", "postprocessing done!, "+ Long.toString(endTime - startTime));

        return bboxes_refined;
    }
}

