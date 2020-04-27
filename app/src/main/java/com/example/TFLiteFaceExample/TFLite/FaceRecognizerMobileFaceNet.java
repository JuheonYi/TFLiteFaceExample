package com.example.TFLiteFaceExample.TFLite;

import android.app.Activity;
import android.util.Log;

import java.io.IOException;

public class FaceRecognizerMobileFaceNet extends FaceRecognizer {

    /**
     * Initializes an {@code ImageClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public FaceRecognizerMobileFaceNet(Activity activity, int h, int w, int c, int feature_dim) throws IOException {
        super(activity, h, w, c, feature_dim);
        Log.d("johnyi-TFLite", "Creating MobileFaceNet");
    }

    @Override
    protected String getModelPath() {
        return "mobileFaceNet.tflite";
    }

    @Override
    protected void runInference(float[][][] input) {
        tflite.run(imgData, OutFeature);
    }
}
