package com.example.facedetproject.UI;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class FaceOverlayView extends View {
    private final Paint paintRect = new Paint();
    private final Paint paintText = new Paint();
    private List<Rect> faces = new ArrayList<>();
    private List<String> names = new ArrayList<>();

    public FaceOverlayView(Context c) { this(c, null); }
    public FaceOverlayView(Context c, AttributeSet a) {
        super(c,a);
        paintRect.setStyle(Paint.Style.STROKE);
        paintRect.setStrokeWidth(6);
        paintRect.setColor(0xFF00FF00);
        paintText.setColor(0xFFFFFFFF);
        paintText.setTextSize(48f);
    }

    public void setFaces(List<Rect> faces, List<String> names) {
        this.faces = faces == null ? new ArrayList<>() : faces;
        this.names = names == null ? new ArrayList<>() : names;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (int i=0; i<faces.size(); i++){
            Rect r = faces.get(i);
            canvas.drawRect(r, paintRect);
            String n = i < names.size() ? names.get(i) : "Unknown";
            canvas.drawText(n, r.left, r.top - 10, paintText);
        }
    }
}
