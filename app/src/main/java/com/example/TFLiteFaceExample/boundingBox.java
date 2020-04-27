package com.example.TFLiteFaceExample;

public class boundingBox{
    public int r0,r1,r2,r3; //width is r0-r2, height is r1-r3
    public int area;
    public float score;
    public boolean has_landmarks = false;
    public int[] landmarks;

    public boundingBox(float _r0, float _r1, float _r2, float _r3, float _score, float[] _landmarks){
        r0 = (int)_r0;
        r1 = (int)_r1;
        r2 = (int)_r2;
        r3 = (int)_r3;
        score = _score;
        area = (r2-r0) * (r3-r1);

        landmarks = new int[10];
        for(int i = 0; i < 10; i++) landmarks[i] = (int)_landmarks[i];
        has_landmarks = true;
    }

    public float calcIOU(boundingBox tmp){
        float IOU = 0;

        int inter_h = (Math.min(r2, tmp.r2) - Math.max(r0, tmp.r0));
        int inter_w = (Math.min(r3, tmp.r3) - Math.max(r1, tmp.r1));

        if(inter_h > 0 && inter_w > 0){
            int intersection = (Math.min(r2, tmp.r2) - Math.max(r0, tmp.r0)) * (Math.min(r3, tmp.r3) - Math.max(r1, tmp.r1));
            IOU = (float) intersection / (area + tmp.area - intersection);
        }
        return IOU;
    }
}