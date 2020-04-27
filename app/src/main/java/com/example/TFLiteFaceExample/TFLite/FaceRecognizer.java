package com.example.TFLiteFaceExample.TFLite;

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;
import android.util.Log;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;

/**
 * Classifies images with Tensorflow Lite.
 */
public abstract class FaceRecognizer {
    /** Tag for the {@link Log}. */
    private static final String TAG = "johnyi-TFLite";

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    int h,w,c,feature_dim;
    float[][] feature;

    /** holds a gpu delegate */
    Delegate gpuDelegate = null;

    /** Initializes an {@code ImageClassifier}. */
    FaceRecognizer(Activity activity, int h, int w, int c, int feature_dim) throws IOException {
        //OutputImgData = new float[1][h*scale][w*scale][1];
        this.h = h;
        this.w = w;
        this.c = c;
        this.feature_dim = feature_dim;

        startTime = SystemClock.uptimeMillis();
        tfliteModel = loadModelFile(activity);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        endTime = SystemClock.uptimeMillis();
        Log.d("johnyi-TFLite", "Created a Tensorflow Lite Face Recognizer. loading time:, "+ Long.toString(endTime - startTime));

        imgData = ByteBuffer.allocateDirect(1 * h * w * c * 4);
        imgData.order(ByteOrder.nativeOrder());

        feature = new float[1][feature_dim];

        OutFeature = ByteBuffer.allocateDirect(feature_dim*4);
        OutFeature.order(ByteOrder.nativeOrder());
        OutFeature.rewind();
        for(int i = 0; i < feature_dim; i ++){}
    }

    public void normalize(float[] dstBuf){
        float norm = 0.0f;
        for(int i = 0; i < feature_dim; i++){
            norm += feature[0][i] * feature[0][i];
        }
        norm = (float)Math.sqrt((double)norm);

        for(int i = 0; i < feature_dim; i++){
            dstBuf[i] = feature[0][i]/norm;
        }
    }

    long startTime, endTime;

    public void extract_feature(byte[] img_in, float[] dstBuf, long[] times){
        imgData.rewind();

        startTime = SystemClock.uptimeMillis();
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                for(int k = 0; k < c; k++){
                    float val = ((float)(img_in[i*w*c + j*c + c-1-k]&0xff)-127.5f)*0.0078125f; //read in BGR scale
                    imgData.putFloat(val);
                }
            }
        }
        endTime = SystemClock.uptimeMillis();
        times[0] = endTime-startTime;

        startTime = SystemClock.uptimeMillis();
        tflite.run(imgData, feature);
        endTime = SystemClock.uptimeMillis();
        times[1] = endTime-startTime;

        startTime = SystemClock.uptimeMillis();
        normalize(dstBuf);
        endTime = SystemClock.uptimeMillis();
        times[2] = endTime-startTime;
    }


    protected ByteBuffer imgData;
    protected ByteBuffer OutFeature;

    public void process(float[][][] input, long[] times) {
        if (tflite == null) Log.e(TAG, "TFLite model has not been initialized; Skipped.");
        imgData.rewind();
        for(int i = 0; i < h; i ++){
            for(int j = 0; j < w; j++){
                imgData.putFloat(0.0f);
            }
        }
        startTime = SystemClock.uptimeMillis();
        runInference(input);
        endTime = SystemClock.uptimeMillis();
        times[1] = endTime-startTime;
        Log.d(TAG, "inference time: " + Long.toString(endTime - startTime));
        OutFeature.rewind();
    }


    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            // TODO(b/120679982)
            // gpuDelegate.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
        if (gpuDelegate == null && GpuDelegateHelper.isGpuDelegateAvailable()) {
            gpuDelegate = GpuDelegateHelper.createGpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }else{
            Log.d("johnyi-TFLite", "GPUDelegate Unavailable!!");
        }
    }

    public void useCPU() {
        tfliteOptions.setUseNNAPI(false);
        recreateInterpreter();
    }

    public void useNNAPI() {
        tfliteOptions.setUseNNAPI(true);
        recreateInterpreter();
    }

    public void setNumThreads(int numThreads) {
        tfliteOptions.setNumThreads(numThreads);
        recreateInterpreter();
    }

    /** Closes tflite to release resources. */
    public void close() {
        tflite.close();
        tflite = null;
        tfliteModel = null;
    }


    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    /**
     * Get the name of the model file stored in Assets.
     *
     * @return
     */
    protected abstract String getModelPath();


    protected abstract void runInference(float[][][] input);
}
