package com.example.attendancefacerecognition.UI;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
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
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
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

    // Collect embeddings for new person
    private final List<float[]> collectedEmbeddings = new ArrayList<>();
    private boolean captureRequested = false;

    private Interpreter faceDetector;


    // -------------------------
    // Load TFLite Model
    // -------------------------
    public static Interpreter loadTFLiteModel(Context context, String modelName) {
        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelName);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();

            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer =
                    fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

            return new Interpreter(modelBuffer);

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // -------------------------
    // onCreate
    // -------------------------
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        previewView = findViewById(R.id.previewViewRegister);
        faceOverlay = findViewById(R.id.faceOverlay);
        btnCapture = findViewById(R.id.btnCapture);
        btnSave = findViewById(R.id.btnSave);
        etName = findViewById(R.id.etName);

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Initialize MediaPipe detector once
        Interpreter faceDetector = Utils.loadBlazeFaceModel(this, "blaze_face_short_range.tflite");
        if (faceDetector == null) {
            Toast.makeText(this, "Failed to load BlazeFace model", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        tflite = loadTFLiteModel(this, "facenet.tflite");
        if (tflite == null) {
            Toast.makeText(this, "Failed to load TFLite model", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        btnCapture.setOnClickListener(v -> {
            captureRequested = true;
            collectedEmbeddings.clear();
            Toast.makeText(this, "Capturing 5 frames. Move face slightly.", Toast.LENGTH_SHORT).show();
        });

        btnSave.setOnClickListener(v -> {
            String name = etName.getText().toString().trim();

            if (name.isEmpty()) {
                Toast.makeText(this, "Enter a name", Toast.LENGTH_SHORT).show();
                return;
            }
            if (collectedEmbeddings.isEmpty()) {
                Toast.makeText(this, "No embeddings collected", Toast.LENGTH_SHORT).show();
                return;
            }

            boolean ok = Utils.appendEmbeddings(this, collectedEmbeddings, name);
            Toast.makeText(this, ok ? "Saved Successfully" : "Failed to Save", Toast.LENGTH_SHORT).show();

            if (ok) finish();
        });

        // Request camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(
                    this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);

        } else startCamera();
    }

    // -------------------------
    // Start Camera
    // -------------------------
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                androidx.camera.core.Preview preview =
                        new androidx.camera.core.Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                ImageAnalysis analysis =
                        new ImageAnalysis.Builder()
                                .setTargetResolution(new Size(640, 480))
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .build();

                analysis.setAnalyzer(cameraExecutor, image -> {
                    Bitmap bitmap = Utils.imageProxyToBitmap(image);
                    if (bitmap == null) {
                        image.close();
                        return;
                    }

                    List<Rect> detectedFaces = Utils.detectFacesBlazeFace(bitmap, faceDetector);
                    runOnUiThread(() -> faceOverlay.setFaces(detectedFaces, null));

                    if (captureRequested && detectedFaces.size() == 1) {
                        Rect rect = detectedFaces.get(0);

                        // Crop safely
                        int left = Math.max(0, rect.left);
                        int top = Math.max(0, rect.top);
                        int right = Math.min(bitmap.getWidth(), rect.right);
                        int bottom = Math.min(bitmap.getHeight(), rect.bottom);
                        if (right - left > 0 && bottom - top > 0) {

                            Bitmap faceBitmap = Bitmap.createBitmap(
                                    bitmap, left, top, right - left, bottom - top);

                            float[] emb = Utils.getFaceEmbedding(faceBitmap, tflite);
                            collectedEmbeddings.add(emb);

                            if (collectedEmbeddings.size() >= 5) {
                                captureRequested = false;
                                runOnUiThread(() ->
                                        Toast.makeText(
                                                this,
                                                "Captured 5 embeddings",
                                                Toast.LENGTH_SHORT
                                        ).show()
                                );
                            }
                        }
                    }

                    image.close();
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis);

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    // -------------------------
    // Permission Result
    // -------------------------
    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_CODE
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

            startCamera();

        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show();
        }
    }


    @Override
    protected void onPause() {
        super.onPause();
        if (cameraExecutor != null) cameraExecutor.shutdownNow();

        try {
            ProcessCameraProvider cameraProvider = ProcessCameraProvider.getInstance(this).get();
            cameraProvider.unbindAll();
        } catch (Exception ignored) {}
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (cameraExecutor != null && !cameraExecutor.isShutdown()) {
            cameraExecutor.shutdownNow();
        }

        try {
            ProcessCameraProvider cameraProvider = ProcessCameraProvider.getInstance(this).get();
            cameraProvider.unbindAll();
        } catch (Exception ignored) {}
    }

}
