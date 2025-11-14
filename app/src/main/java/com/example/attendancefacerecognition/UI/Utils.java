package com.example.attendancefacerecognition.UI;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import org.json.JSONArray;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    private static final String TAG = "Utils";

    // ========================================
    // Load BlazeFace TFLite model
    // ========================================
    public static Interpreter loadBlazeFaceModel(Context context, String modelName) {
        try {
            FileInputStream fis = new FileInputStream(context.getAssets().openFd(modelName).getFileDescriptor());
            FileChannel fileChannel = fis.getChannel();
            long startOffset = context.getAssets().openFd(modelName).getStartOffset();
            long declaredLength = context.getAssets().openFd(modelName).getDeclaredLength();
            MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            fis.close();
            return new Interpreter(buffer);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // ========================================
    // Detect faces with BlazeFace
    // ========================================
    public static List<Rect> detectFacesBlazeFace(Bitmap bitmap, Interpreter interpreter) {
        List<Rect> faces = new ArrayList<>();
        if (interpreter == null || bitmap == null) return faces;

        try {
            int inputSize = 128; // BlazeFace input size
            Bitmap scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);

            ByteBuffer input = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4);
            input.order(ByteOrder.nativeOrder());

            int[] pixels = new int[inputSize * inputSize];
            scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize);

            for (int pixel : pixels) {
                input.putFloat(((pixel >> 16) & 0xFF) / 255.f);
                input.putFloat(((pixel >> 8) & 0xFF) / 255.f);
                input.putFloat((pixel & 0xFF) / 255.f);
            }
            input.rewind();

            float[][][] output = new float[1][896][16];
            interpreter.run(input, output);

            for (int i = 0; i < 896; i++) {
                float score = output[0][i][4]; // confidence score
                if (score > 0.5f) {
                    float xMin = output[0][i][0] * bitmap.getWidth();
                    float yMin = output[0][i][1] * bitmap.getHeight();
                    float xMax = output[0][i][2] * bitmap.getWidth();
                    float yMax = output[0][i][3] * bitmap.getHeight();

                    faces.add(new Rect(
                            Math.max(0, Math.round(xMin)),
                            Math.max(0, Math.round(yMin)),
                            Math.min(bitmap.getWidth(), Math.round(xMax)),
                            Math.min(bitmap.getHeight(), Math.round(yMax))
                    ));
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Face detection failed: " + e.getMessage());
        }

        return faces;
    }

    // ========================================
    // Convert ImageProxy to Bitmap
    // ========================================
    public static Bitmap imageProxyToBitmap(ImageProxy image) {
        try {
            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
            ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();

            byte[] nv21 = new byte[ySize + uSize + vSize];

            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);

            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, image.getWidth(), image.getHeight()), 100, out);

            Bitmap bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size());

            Matrix matrix = new Matrix();
            matrix.postRotate(image.getImageInfo().getRotationDegrees());

            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // ========================================
    // Get FaceNet embedding (512-d)
    // ========================================
    public static float[] getFaceEmbedding(Bitmap faceBitmap, Interpreter tflite) {
        Bitmap scaled = Bitmap.createScaledBitmap(faceBitmap, 160, 160, true);
        ByteBuffer inputBuffer = convertBitmapToBuffer(scaled);

        float[][] embeddings = new float[1][512]; // FaceNet output
        tflite.run(inputBuffer, embeddings);

        // Normalize
        float norm = 0f;
        for (float v : embeddings[0]) norm += v * v;
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < 512; i++) embeddings[0][i] /= norm;

        return embeddings[0];
    }

    private static ByteBuffer convertBitmapToBuffer(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(1 * 160 * 160 * 3 * 4);
        buffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[160 * 160];
        bitmap.getPixels(pixels, 0, 160, 0, 0, 160, 160);
        for (int pixel : pixels) {
            buffer.putFloat(((pixel >> 16) & 0xFF) / 255.f);
            buffer.putFloat(((pixel >> 8) & 0xFF) / 255.f);
            buffer.putFloat((pixel & 0xFF) / 255.f);
        }
        buffer.rewind();
        return buffer;
    }

    // ========================================
    // Recognize face
    // ========================================
    public static String recognizeFace(float[] emb, float[][] knownEmbeddings, List<String> knownNames, float threshold) {
        String bestName = "Unknown";
        float minDist = Float.MAX_VALUE;

        for (int i = 0; i < knownEmbeddings.length; i++) {
            if (knownEmbeddings[i].length != emb.length) continue; // safety
            float dist = l2Distance(emb, knownEmbeddings[i]);
            if (dist < minDist) {
                minDist = dist;
                bestName = knownNames.get(i);
            }
        }

        return (minDist < threshold) ? bestName : "Unknown";
    }

    private static float l2Distance(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Embedding size mismatch");
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    // ========================================
    // Save/load embeddings and names
    // ========================================
    public static boolean appendEmbeddings(Context context, List<float[]> newEmbeddings, String name) {
        try {
            List<String> names = loadNames(context);
            names.add(name);

            List<float[]> embeddingsList = loadEmbeddings(context);
            embeddingsList.addAll(newEmbeddings);

            saveNames(context, names);
            saveEmbeddings(context, embeddingsList);

            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static void saveNames(Context context, List<String> names) throws IOException {
        JSONArray json = new JSONArray();
        for (String n : names) json.put(n);
        FileOutputStream fos = context.openFileOutput("names.json", Context.MODE_PRIVATE);
        fos.write(json.toString().getBytes());
        fos.close();
    }

    private static List<String> loadNames(Context context) {
        List<String> names = new ArrayList<>();
        try {
            FileInputStream fis = context.openFileInput("names.json");
            byte[] data = new byte[fis.available()];
            fis.read(data);
            fis.close();

            JSONArray json = new JSONArray(new String(data));
            for (int i = 0; i < json.length(); i++) names.add(json.getString(i));
        } catch (Exception ignore) {}
        return names;
    }

    private static void saveEmbeddings(Context context, List<float[]> embeddings) throws IOException {
        FileOutputStream fos = context.openFileOutput("embeddings.bin", Context.MODE_PRIVATE);
        for (float[] emb : embeddings) {
            if (emb.length != 512) continue; // safety check
            ByteBuffer buffer = ByteBuffer.allocate(512 * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : emb) buffer.putFloat(v);
            fos.write(buffer.array());
        }
        fos.close();
    }

    private static List<float[]> loadEmbeddings(Context context) {
        List<float[]> embeddings = new ArrayList<>();
        try {
            FileInputStream fis = context.openFileInput("embeddings.bin");
            byte[] data = readAllBytesCompat(fis);
            fis.close();

            ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            while (buffer.remaining() >= 512 * 4) {
                float[] emb = new float[512];
                for (int i = 0; i < 512; i++) emb[i] = buffer.getFloat();
                embeddings.add(emb);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return embeddings;
    }

    public static byte[] readAllBytesCompat(InputStream is) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] data = new byte[4096];
        int nRead;
        while ((nRead = is.read(data)) != -1) buffer.write(data, 0, nRead);
        return buffer.toByteArray();
    }
}
