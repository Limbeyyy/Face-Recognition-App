package com.example.attendancefacerecognition.UI;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.attendancefacerecognition.R;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AttendanceActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_CODE = 101;

    private PreviewView previewView;
    private FaceOverlayView faceOverlay;
    private ExecutorService cameraExecutor;
    private Interpreter tflite;

    // aggregation
    private final List<String> frameResults = new ArrayList<>();
    private final Handler handler = new Handler();
    private Runnable aggregateRunnable;

    // known embeddings/names loaded from assets
    private float[][] knownEmbeddings;
    private String[] knownNames;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_attendance);

        previewView = findViewById(R.id.previewView);
        faceOverlay = findViewById(R.id.faceOverlay);
        cameraExecutor = Executors.newSingleThreadExecutor();

        try {
            tflite = new Interpreter(Utils.loadModelFile(this, "facenet.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Model load failed", Toast.LENGTH_SHORT).show();
            return;
        }

        // load known embeddings & names from assets
        Utils.EmbeddingsData data = Utils.loadEmbeddingsAndNames(this, "embeddings.bin", "names.json");
        knownEmbeddings = data.embeddings;
        knownNames = data.names;

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_CODE);
        } else {
            startCamera();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                androidx.camera.core.Preview preview = new androidx.camera.core.Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(cameraExecutor, image -> {
                    processImageProxy(image);
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis);

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void processImageProxy(@NonNull ImageProxy image) {
        Bitmap bitmap = Utils.imageProxyToBitmap(image);
        if (bitmap != null) {
            // 1) detect faces using MediaPipe (TODO: implement in Utils)
            List<Rect> faces = Utils.detectFacesMediaPipe(bitmap);

            // 2) For each face, crop, get embedding & match
            List<String> namesForOverlay = new ArrayList<>();
            for (Rect r : faces) {
                // safe crop
                int left = Math.max(0, r.left);
                int top = Math.max(0, r.top);
                int right = Math.min(bitmap.getWidth(), r.right);
                int bottom = Math.min(bitmap.getHeight(), r.bottom);
                if (right - left <= 0 || bottom - top <= 0) continue;

                Bitmap faceBmp = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top);
                float[] emb = Utils.getFaceEmbedding(faceBmp, tflite);
                String name = Utils.recognizeFace(emb, knownEmbeddings, knownNames, 0.65f);
                namesForOverlay.add(name);
                frameResults.add(name);
            }

            // update overlay (UI thread)
            runOnUiThread(() -> faceOverlay.setFaces(faces, namesForOverlay));

            // aggregation: schedule result after 5 seconds of inactivity
            if (aggregateRunnable != null) handler.removeCallbacks(aggregateRunnable);
            aggregateRunnable = () -> {
                String confirmed = getMostFrequent(frameResults);
                runOnUiThread(() -> Toast.makeText(AttendanceActivity.this, "Marked: " + confirmed, Toast.LENGTH_LONG).show());
                frameResults.clear();
                // TODO: Save attendance (send to server or local DB)
                // optionally finish() or show UI
            };
            handler.postDelayed(aggregateRunnable, 5000);
        }
        image.close();
    }

    private String getMostFrequent(List<String> list) {
        Map<String, Integer> freq = new HashMap<>();
        for (String s : list) freq.put(s, freq.getOrDefault(s, 0) + 1);
        String best = "Unknown"; int bestCount = 0;
        for (Map.Entry<String, Integer> e : freq.entrySet()) {
            if (e.getValue() > bestCount) { bestCount = e.getValue(); best = e.getKey(); }
        }
        return best;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,@NonNull String[] permissions,@NonNull int[] grantResults){
        super.onRequestPermissionsResult(requestCode,permissions,grantResults);
        if (requestCode==CAMERA_PERMISSION_CODE && grantResults.length>0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
            startCamera();
        } else {
            Toast.makeText(this,"Camera permission denied",Toast.LENGTH_SHORT).show();
        }
    }
}