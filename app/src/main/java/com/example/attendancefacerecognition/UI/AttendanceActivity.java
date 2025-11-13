package com.example.attendancefacerecognition.UI;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.example.attendancefacerecognition.R;
import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONArray;
import org.json.JSONException;
import org.tensorflow.lite.Interpreter;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AttendanceActivity extends AppCompatActivity {

    private PreviewView previewView;
    private FaceOverlayView faceOverlay;
    private Button btnSwitchCamera;

    private ExecutorService cameraExecutor;
    private Handler handler = new Handler();
    private Runnable aggregateRunnable;

    private boolean useFrontCamera = true;

    private List<String> knownNames;
    private List<float[]> knownEmbeddings;
    private Interpreter tflite;

    private List<String> frameResults = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_attendance);

        previewView = findViewById(R.id.previewView);
        faceOverlay = findViewById(R.id.faceOverlay);
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera);

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Load names, embeddings, model
        knownNames = loadNamesFromJson("names.json");
        knownEmbeddings = loadEmbeddings("embeddings.bin");
        tflite = loadModelFile("facenet.tflite");

        btnSwitchCamera.setOnClickListener(v -> {
            useFrontCamera = !useFrontCamera;
            startCamera();
        });

        startCamera();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                CameraSelector cameraSelector = useFrontCamera ?
                        CameraSelector.DEFAULT_FRONT_CAMERA :
                        CameraSelector.DEFAULT_BACK_CAMERA;

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(cameraExecutor, this::processImageProxy);

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
            List<Rect> detectedFaces = Utils.detectFacesMediaPipe(bitmap);
            List<String> namesForOverlay = new ArrayList<>();
            List<Rect> scaledRects = new ArrayList<>();

            float scaleX = previewView.getWidth() / (float) bitmap.getWidth();
            float scaleY = previewView.getHeight() / (float) bitmap.getHeight();

            for (Rect r : detectedFaces) {
                int left = Math.max(0, r.left);
                int top = Math.max(0, r.top);
                int right = Math.min(bitmap.getWidth(), r.right);
                int bottom = Math.min(bitmap.getHeight(), r.bottom);
                if (right - left <= 0 || bottom - top <= 0) continue;

                Bitmap faceBmp = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top);
                float[] emb = Utils.getFaceEmbedding(faceBmp, tflite);

                float[][] embeddingsArray = new float[knownEmbeddings.size()][128];
                for (int i = 0; i < knownEmbeddings.size(); i++) embeddingsArray[i] = knownEmbeddings.get(i);

                String name = Utils.recognizeFace(emb, embeddingsArray, knownNames, 0.65f);
                namesForOverlay.add(name);
                frameResults.add(name);

                Rect scaled = new Rect(
                        (int) (left * scaleX),
                        (int) (top * scaleY),
                        (int) (right * scaleX),
                        (int) (bottom * scaleY)
                );
                scaledRects.add(scaled);
            }

            runOnUiThread(() -> faceOverlay.setFaces(scaledRects, namesForOverlay));

            // Aggregate every 5 seconds
            if (aggregateRunnable != null) handler.removeCallbacks(aggregateRunnable);
            aggregateRunnable = () -> {
                String confirmed = getMostFrequent(frameResults);
                runOnUiThread(() -> Toast.makeText(this,
                        "Attendance Marked: " + confirmed, Toast.LENGTH_LONG).show());
                frameResults.clear();
            };
            handler.postDelayed(aggregateRunnable, 5000);
        }
        image.close();
    }

    private String getMostFrequent(List<String> list) {
        if (list.isEmpty()) return "Unknown";
        String most = null;
        int maxCount = 0;
        for (String s : list) {
            int count = 0;
            for (String t : list) if (t.equals(s)) count++;
            if (count > maxCount) {
                maxCount = count;
                most = s;
            }
        }
        return most;
    }

    private List<String> loadNamesFromJson(String fileName) {
        List<String> names = new ArrayList<>();
        try {
            InputStream is = getAssets().open(fileName);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            String json = new String(buffer, StandardCharsets.UTF_8);
            JSONArray jsonArray = new JSONArray(json);
            for (int i = 0; i < jsonArray.length(); i++) names.add(jsonArray.getString(i));
        } catch (IOException | JSONException e) {
            e.printStackTrace();
        }
        return names;
    }

    private List<float[]> loadEmbeddings(String fileName) {
        List<float[]> embeddings = new ArrayList<>();
        try {
            InputStream is = getAssets().open(fileName);
            DataInputStream dis = new DataInputStream(is);
            while (dis.available() > 0) {
                float[] embedding = new float[128];
                for (int i = 0; i < 128; i++) embedding[i] = dis.readFloat();
                embeddings.add(embedding);
            }
            dis.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return embeddings;
    }

    private Interpreter loadModelFile(String modelFile) {
        try {
            AssetFileDescriptor fileDescriptor = getAssets().openFd(modelFile);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            return new Interpreter(modelBuffer);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
