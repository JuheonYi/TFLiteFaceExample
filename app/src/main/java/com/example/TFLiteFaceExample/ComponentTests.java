package com.example.TFLiteFaceExample;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.TFLiteFaceExample.TFLite.FaceDetectorRetina;
import com.example.TFLiteFaceExample.TFLite.FaceRecognizer;
import com.example.TFLiteFaceExample.TFLite.FaceRecognizerMobileFaceNet;
import com.example.TFLiteFaceExample.TFLite.FaceDetector;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Vector;

public class ComponentTests extends AppCompatActivity{
    //load OpenCV
    static {
        try {
            System.loadLibrary("opencv_java3");
            Log.d("johnyi-jni", "Successfully loaded OpenCV");
        } catch (UnsatisfiedLinkError e) {
            Log.d("johnyi-jni", "OpenCV not found");
        }
    }

    private FaceDetector facedetector;
    private FaceRecognizer facerecognizer;

    private static final int REQUEST_CODE = 1;

    private Activity a;

    //for detection
    Bitmap bmpOrig;
    ImageView imageView_original;
    Mat matOrig, matDisplay;
    int H, W;

    int img_idx = 0;
    int probe_idx1 = 0;
    int probe_idx2 = 0;

    //for recognition
    ImageView imageView_probe1, imageView_probe2;
    Bitmap bmpProbe1, bmpProbe2;
    Mat matProbe1, matProbe2;
    byte[] probe1_bytes, probe2_bytes;
    float[] probe1_feature, probe2_feature;
    int H_probe = 112;
    int W_probe = 112;
    int feature_dim = 512;

    int numFrames = 3;

    String dir_name = "scenes";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.component_test);
        requestPermissions();

        a = this;
        //Face detection related block
        imageView_original = findViewById(R.id.original);
        img_idx = 0;
        bmpOrig = utils.getBitmapFromAssets(this, String.format("%s/%04d.png", dir_name, img_idx));

        H = bmpOrig.getHeight();
        W = bmpOrig.getWidth();
        matOrig = new Mat(H, W, CvType.CV_8UC3);

        Utils.bitmapToMat(bmpOrig , matOrig);
        Imgproc.cvtColor(matOrig, matOrig, Imgproc.COLOR_RGBA2RGB);

        matDisplay = matOrig.clone();

        imageView_original.setImageBitmap(bmpOrig);

        FaceDetectionResult = (TextView)findViewById(R.id.txt_face_detection_result);

        FaceRecognitionResult = (TextView)findViewById(R.id.txt_face_recognition_result);
        imageView_probe1 = findViewById(R.id.probe1);
        imageView_probe2 = findViewById(R.id.probe2);
        bmpProbe1 = utils.getBitmapFromAssets(this, "probes/same1-0.png");
        matProbe1 = new Mat();
        Utils.bitmapToMat(bmpProbe1 , matProbe1);
        Imgproc.cvtColor(matProbe1, matProbe1, Imgproc.COLOR_RGBA2RGB);
        probe1_bytes = new byte[H_probe * W_probe * matProbe1.channels()];
        matProbe1.clone().get(0,0, probe1_bytes);

        bmpProbe2 = utils.getBitmapFromAssets(this, "probes/same2-0.png");
        matProbe2 = new Mat();
        Utils.bitmapToMat(bmpProbe2 , matProbe2);
        Imgproc.cvtColor(matProbe2, matProbe2, Imgproc.COLOR_RGBA2RGB);
        probe2_bytes = new byte[H_probe * W_probe * matProbe2.channels()];
        matProbe2.clone().get(0,0, probe2_bytes);

        imageView_probe1.setImageBitmap(bmpProbe1);
        imageView_probe2.setImageBitmap(bmpProbe2);
        imageView_probe1.setScaleType(ImageView.ScaleType.FIT_XY);
        imageView_probe2.setScaleType(ImageView.ScaleType.FIT_XY);

        probe1_feature = new float[feature_dim];
        probe2_feature = new float[feature_dim];

        try {
            facerecognizer = new FaceRecognizerMobileFaceNet(a, H_probe, W_probe, 3, feature_dim);
            facerecognizer.useGpu();
        }catch(Exception e){
            Log.d("johnyi-TFLite", "Exception: "+e);
        }
    }

    public void onTestFaceRecognizer(View view){
        //extract features
        try {
            facerecognizer.extract_feature(probe1_bytes, probe1_feature, times);
            facerecognizer.extract_feature(probe2_bytes, probe2_feature, times);

            final float distance = utils.l2_distance(probe1_feature, probe2_feature);

            runOnUiThread(new Runnable() {
                public void run() {
                    FaceRecognitionResult.setText(String.format("L2 distance: %.4f inference time: %d ms", distance, (int) times[1]));
                }
            });
        }catch(Exception e){
            Log.d("johnyi-TFLite", "Exception: "+e);
        }
    }

    public void onSwitchProbe(View view){
        switch (view.getId()) {
            case R.id.bt_switchProbe1: {
                probe_idx1 = (probe_idx1+1)%10;
                bmpProbe1 = utils.getBitmapFromAssets(this, String.format("probes/same1-%d.png", probe_idx1));
                imageView_probe1.setImageBitmap(bmpProbe1);
                Utils.bitmapToMat(bmpProbe1 , matProbe1);
                Imgproc.cvtColor(matProbe1, matProbe1, Imgproc.COLOR_RGBA2RGB);
                probe1_bytes = new byte[H_probe * W_probe * matProbe1.channels()];
                matProbe1.clone().get(0,0, probe1_bytes);

                bmpProbe2 = utils.getBitmapFromAssets(this, String.format("probes/same2-%d.png", probe_idx1));
                imageView_probe2.setImageBitmap(bmpProbe2);
                Utils.bitmapToMat(bmpProbe2 , matProbe2);
                Imgproc.cvtColor(matProbe2, matProbe2, Imgproc.COLOR_RGBA2RGB);
                probe2_bytes = new byte[H_probe * W_probe * matProbe2.channels()];
                matProbe2.clone().get(0,0, probe2_bytes);
                break;
            }
            case R.id.bt_switchProbe2: {
                probe_idx2 = (probe_idx2+1)%10;
                bmpProbe1 = utils.getBitmapFromAssets(this, String.format("probes/diff1-%d.png", probe_idx2));
                imageView_probe1.setImageBitmap(bmpProbe1);
                Utils.bitmapToMat(bmpProbe1 , matProbe1);
                Imgproc.cvtColor(matProbe1, matProbe1, Imgproc.COLOR_RGBA2RGB);
                probe1_bytes = new byte[H_probe * W_probe * matProbe1.channels()];
                matProbe1.clone().get(0,0, probe1_bytes);

                bmpProbe2 = utils.getBitmapFromAssets(this, String.format("probes/diff2-%d.png", probe_idx2));
                imageView_probe2.setImageBitmap(bmpProbe2);
                Utils.bitmapToMat(bmpProbe2 , matProbe2);
                Imgproc.cvtColor(matProbe2, matProbe2, Imgproc.COLOR_RGBA2RGB);
                probe2_bytes = new byte[H_probe * W_probe * matProbe2.channels()];
                matProbe2.clone().get(0,0, probe2_bytes);
                break;
            }
        }
    }

    public void onResetDetectionImage(View view){
        imageView_original.setImageBitmap(bmpOrig);
        runOnUiThread(new Runnable() {
            public void run() {
                FaceDetectionResult.setText("Inference time: ______ ms");
            }
        });
    }

    public void onSwitchDetectionImage(View view){
        img_idx = (img_idx+1)%numFrames;
        bmpOrig = utils.getBitmapFromAssets(this, String.format("%s/%04d.png", dir_name, img_idx));
        H = bmpOrig.getHeight();
        W = bmpOrig.getWidth();
        matOrig = new Mat(H, W, CvType.CV_8UC3);
        Utils.bitmapToMat(bmpOrig , matOrig);
        Imgproc.cvtColor(matOrig, matOrig, Imgproc.COLOR_RGBA2RGB);
        Log.d("johnyi","channels: "+matOrig.channels());

        imageView_original.setImageBitmap(bmpOrig);

        matDisplay = matOrig.clone();
        runOnUiThread(new Runnable() {
            public void run() {
                FaceDetectionResult.setText("Inference time: ______ ms");
            }
        });
    }

    TextView FaceDetectionResult, FaceRecognitionResult;

    Thread testThread;
    long[] times = new long[3];

    long startTime, endTime;
    public void onTestFaceDetectorRetina(View view){
        testThread = new Thread(new Runnable() {
            @Override
            public void run() {
                if(facedetector != null){
                    facedetector.close();
                    facedetector = null;
                }
                try {
                    int h = 1080;
                    int w = 1920;
                    int c = 3;
                    byte[] img_bytes = new byte[h*w*c];

                    Mat matResized = new Mat();
                    Imgproc.resize(matOrig, matResized, new Size(w,h),0,0, Imgproc.INTER_CUBIC);
                    Log.d("johnyi-TFLite",String.format("resized Rows x Cols:%dx%d", matResized.rows(),matResized.cols()));
                    matResized.clone().get(0,0, img_bytes);

                    facedetector = new FaceDetectorRetina(a, h,w,c, 1f,"MobileNet");
                    //facedetector.useGpu();

                    times[0] = 0;
                    times[1] = 0;
                    times[2] = 0;

                    Log.d("johnyi","start detection!!!");
                    startTime = SystemClock.uptimeMillis();
                    Vector<boundingBox> bboxes = facedetector.detect(img_bytes, times);
                    Log.d("johnyi",String.format("detected %d faces!!!",bboxes.size()));
                    endTime = SystemClock.uptimeMillis();
                    runOnUiThread(new Runnable() {
                        public void run() {
                            FaceDetectionResult.setText(String.format("Inference time: %d ms", (int)times[1]));
                        }
                    });

                    utils.overlay_bboxes(matResized, bboxes);
                    Imgproc.resize(matResized, matResized, new Size(W,H),0,0, Imgproc.INTER_CUBIC);

                    Bitmap bmpOut = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(matResized, bmpOut);
                    imageView_original.setImageBitmap(bmpOut);
                }catch (Exception e){
                    Log.d("johnyi-TFLite","initialization Exception: "+e);
                }
            }
        });
        testThread.start();
    }


    @Override
    public void onResume() {
        super.onResume();
    }

    private void requestPermissions() {
        String[] PERMISSIONS = {
                android.Manifest.permission.RECORD_AUDIO,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE
        };

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(PERMISSIONS, REQUEST_CODE);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
    }
}
