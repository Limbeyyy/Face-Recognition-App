package com.example.attendancefacerecognition.UI;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult;

import org.tensorflow.lite.Interpreter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    private static final String TAG = "Utils";
    private static FaceDetector faceDetector;
    private static Context context;

    // 🔹 Initialize MediaPipe Face Detector once
    public static void initFaceDetector(Context context) {
        if (faceDetector != null) return;
        try {
            FaceDetector.FaceDetectorOptions options = FaceDetector.FaceDetectorOptions.builder()
                    .setMinDetectionConfidence(0.65f)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .build();
            faceDetector = FaceDetector.createFromOptions(context, options);
            Log.d(TAG, "✅ FaceDetector initialized");
        } catch (Exception e) {
            Log.e(TAG, "❌ Failed to init FaceDetector: " + e.getMessage());
        }
    }

    // 🔹 Convert CameraX ImageProxy to Bitmap
    public static Bitmap imageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);

        Bitmap bitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(bytes));

        // Rotate if needed
        Matrix matrix = new Matrix();
        matrix.postRotate(image.getImageInfo().getRotationDegrees());
        Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        bitmap.recycle();
        return rotated;
    }

    // 🔹 Detect faces using MediaPipe
    public static List<Rect> detectFacesMediaPipe(Bitmap bitmap) {
        List<Rect> faces = new ArrayList<>();
        try {
            // Initialize MediaPipe face detector once
            if (faceDetector == null) {
                FaceDetector.FaceDetectorOptions options = FaceDetector.FaceDetectorOptions.builder()
                        .setRunningMode(RunningMode.IMAGE)
                        .setMinDetectionConfidence(0.65f)
                        .build();
                faceDetector = FaceDetector.createFromOptions(context, options);
            }

            MPImage mpImage = new BitmapImageBuilder(bitmap).build();
            FaceDetectorResult result = faceDetector.detect(mpImage);

            if (result != null && !result.detections().isEmpty()) {
                for (Detection detection : result.detections()) {
                    RectF rectF = detection.boundingBox();
                    if (rectF != null) {
                        faces.add(new Rect(
                                Math.round(rectF.left),
                                Math.round(rectF.top),
                                Math.round(rectF.right),
                                Math.round(rectF.bottom)
                        ));
                    }
                }
            }
        } catch (Exception e) {
            Log.e("Utils", "Face detection failed: " + e.getMessage());
        }
        return faces;
    }


    // 🔹 Get Face Embedding using TFLite model
    public static float[] getFaceEmbedding(Bitmap faceBitmap, Interpreter tflite) {
        Bitmap scaled = Bitmap.createScaledBitmap(faceBitmap, 160, 160, true);
        ByteBuffer inputBuffer = convertBitmapToBuffer(scaled);

        float[][] embeddings = new float[1][128];
        tflite.run(inputBuffer, embeddings);

        // Normalize the embedding
        float norm = 0f;
        for (float v : embeddings[0]) norm += v * v;
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < 128; i++) embeddings[0][i] /= norm;

        return embeddings[0];
    }

    // 🔹 Helper to convert bitmap to float buffer
    private static ByteBuffer convertBitmapToBuffer(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(1 * 160 * 160 * 3 * 4);
        buffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[160 * 160];
        bitmap.getPixels(pixels, 0, 160, 0, 0, 160, 160);
        for (int pixel : pixels) {
            buffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f);
            buffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);
            buffer.putFloat((pixel & 0xFF) / 255.0f);
        }
        buffer.rewind();
        return buffer;
    }

    // 🔹 Recognize Face (compare embeddings)
    public static String recognizeFace(float[] emb, float[][] knownEmbeddings, List<String> knownNames, float threshold) {
        String bestName = "Unknown";
        float minDist = Float.MAX_VALUE;

        for (int i = 0; i < knownEmbeddings.length; i++) {
            float dist = l2Distance(emb, knownEmbeddings[i]);
            if (dist < minDist) {
                minDist = dist;
                bestName = knownNames.get(i);
            }
        }
        return (minDist < threshold) ? bestName : "Unknown";
    }

    // 🔹 Calculate L2 distance
    private static float l2Distance(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }
}
