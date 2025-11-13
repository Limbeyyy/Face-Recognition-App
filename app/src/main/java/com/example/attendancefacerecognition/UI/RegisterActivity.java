package com.example.attendancefacerecognition.UI;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.EditText;
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
import com.example.attendancefacerecognition.UI.Utils;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;



public class RegisterActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_CODE = 111;
    private PreviewView previewView;
    private Button btnCapture, btnSave;
    private EditText etName;
    private ExecutorService cameraExecutor;
    private Interpreter tflite;

    private FaceOverlayView faceOverlay;

    private float[][] knownEmbeddings;
    private String[] knownNames;
    private final List<String> frameResults = new ArrayList<>();
    private final Handler handler = new Handler();
    private Runnable aggregateRunnable;

    // collect embeddings for the new person
    private List<float[]> collectedEmbeddings = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        previewView = findViewById(R.id.previewViewRegister);
        faceOverlay = findViewById(R.id.faceOverlay);
        btnCapture = findViewById(R.id.btnCapture);
        btnSave = findViewById(R.id.btnSave);
        etName = findViewById(R.id.etName);

        try {
            tflite = new Interpreter(Utils.getFaceEmbedding(this, "facenet.tflite"));
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
        }

        btnCapture.setOnClickListener(v -> Toast.makeText(this, "Capturing 5 frames; move head slightly", Toast.LENGTH_SHORT).show());
        btnSave.setOnClickListener(v -> {
            String name = etName.getText().toString().trim();
            if (name.isEmpty()) { Toast.makeText(this,"Enter name",Toast.LENGTH_SHORT).show(); return; }
            if (collectedEmbeddings.isEmpty()) { Toast.makeText(this,"No captures",Toast.LENGTH_SHORT).show(); return; }
            boolean ok = Utils.appendEmbeddings(this, collectedEmbeddings, name);
            Toast.makeText(this, ok ? "Saved":"Save failed", Toast.LENGTH_SHORT).show();
            if (ok) finish();
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        } else startCamera();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Preview use case
                androidx.camera.core.Preview preview = new androidx.camera.core.Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Front camera
                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                // Image analysis use case
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(cameraExecutor, image -> {
                    try {
                        Bitmap bitmap = Utils.imageProxyToBitmap(image);
                        if (bitmap != null) {

                            // Step 1: Detect faces using MediaPipe (you can replace with ML Kit if needed
                            Log.d("AttendanceActivity", "Processing bitmap w=" + bitmap.getWidth() + " h=" + bitmap.getHeight());

                            List<Rect> faces = Utils.detectFacesMediaPipe(bitmap);
                            Log.d("AttendanceActivity", "Detected faces count: " + faces.size());

                            // Step 2: For each detected face, crop and compute embeddings
                            List<String> detectedNames = new ArrayList<>();
                            for (Rect rect : faces) {
                                // Crop safely
                                int left = Math.max(0, rect.left);
                                int top = Math.max(0, rect.top);
                                int right = Math.min(bitmap.getWidth(), rect.right);
                                int bottom = Math.min(bitmap.getHeight(), rect.bottom);
                                if (right - left <= 0 || bottom - top <= 0) continue;

                                Bitmap faceBitmap = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top);

                                // Get face embedding
                                float[] embedding = Utils.getFaceEmbedding(faceBitmap, tflite);

                                // Compare embedding with stored known embeddings
                                String recognizedName = Utils.recognizeFace(
                                        embedding,
                                        knownEmbeddings,
                                        Arrays.asList(knownNames),
                                        0.65f
                                );

                                detectedNames.add(recognizedName);
                            }

                            // Step 3: Show detected faces on overlay
                            runOnUiThread(() -> faceOverlay.setFaces(faces, detectedNames));

                            // Step 4: Aggregate or mark attendance (optional)
                            if (!detectedNames.isEmpty()) {
                                frameResults.addAll(detectedNames);
                                if (aggregateRunnable != null) handler.removeCallbacks(aggregateRunnable);
                                aggregateRunnable = () -> {
                                    String confirmed = getMostFrequent(frameResults);
                                    runOnUiThread(() ->
                                            Toast.makeText(
                                                    RegisterActivity.this,
                                                    "Marked Attendance: " + confirmed,
                                                    Toast.LENGTH_LONG
                                            ).show()
                                    );
                                    frameResults.clear();
                                };
                                handler.postDelayed(aggregateRunnable, 4000);
                            }
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        image.close(); // Always close image or camera will freeze
                    }
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis);

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private String getMostFrequent(List<String> names) {
        if (names.isEmpty()) return "Unknown";

        Map<String, Integer> freqMap = new HashMap<>();
        for (String name : names) {
            freqMap.put(name, freqMap.getOrDefault(name, 0) + 1);
        }

        return Collections.max(freqMap.entrySet(), Map.Entry.comparingByValue()).getKey();
    }


    // handle permission result
    @Override
    public void onRequestPermissionsResult(int requestCode,@NonNull String[] permissions,@NonNull int[] grantResults){
        super.onRequestPermissionsResult(requestCode,permissions,grantResults);
        if(requestCode==CAMERA_PERMISSION_CODE && grantResults.length>0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
            startCamera();
        } else {
            Toast.makeText(this,"Camera permission required",Toast.LENGTH_SHORT).show();
        }
    }
}
