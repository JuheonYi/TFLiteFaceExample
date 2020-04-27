package com.example.TFLiteFaceExample;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.InputStream;
import java.util.Vector;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class utils {
    public static float l2_distance(float[] feat1, float[] feat2){
        float distance = 0;
        for(int i = 0; i < feat1.length; i++){
            distance += Math.pow((double)(feat1[i]-feat2[i]),2);
        }
        return distance;
    }

    public static void overlay_bboxes(Mat img_mat, Vector<boundingBox> bboxes){
        boundingBox tmp;
        for(int i = 0; i < bboxes.size(); i++){
            tmp = bboxes.elementAt(i);

            Imgproc.rectangle(img_mat, new Point(tmp.r0, tmp.r1), new Point(tmp.r2, tmp.r3), new Scalar(255, 0, 0, 255), 3);

            // display landmarks
            if (tmp.has_landmarks) {
                for (int l = 0; l < 5; l++) {
                    Imgproc.circle(img_mat, new Point(tmp.landmarks[2 * l], tmp.landmarks[2 * l + 1]), 1, new Scalar(0, 255, 0), 2);
                }
            }
        }
    }

    public static Bitmap getBitmapFromAssets(Activity activity, String fileName) {
        AssetManager assetManager = activity.getAssets();
        Bitmap bitmap;
        try {
            InputStream istr = assetManager.open(fileName);
            bitmap = BitmapFactory.decodeStream(istr);
        }catch(Exception e){
            Log.d("johnyi-utils","Exception "+e);
            bitmap = Bitmap.createBitmap(224,224, Bitmap.Config.ARGB_8888);
        }
        return bitmap;
    }
}