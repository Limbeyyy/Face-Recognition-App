package com.example.facedetproject.UI;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
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

import com.example.facedetproject.R;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

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

    // collect embeddings for the new person
    private List<float[]> collectedEmbeddings = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        previewView = findViewById(R.id.previewViewRegister);
        btnCapture = findViewById(R.id.btnCapture);
        btnSave = findViewById(R.id.btnSave);
        etName = findViewById(R.id.etName);

        cameraExecutor = Executors.newSingleThreadExecutor();

        try {
            tflite = new Interpreter(Utils.loadModelFile(this, "facenet.tflite"));
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
                androidx.camera.core.Preview preview = new androidx.camera.core.Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(cameraExecutor, image -> {
                    // When user presses capture, we capture N frames and compute embeddings
                    // For simplicity, only compute embedding on demand; actual capture logic
                    // can store frames into a small buffer to pick next N frames.
                    image.close();
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
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
