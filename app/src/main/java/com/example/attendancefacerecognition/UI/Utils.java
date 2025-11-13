package com.example.attendancefacerecognition.UI;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.util.Log;
import android.util.Pair;

import androidx.camera.core.ImageProxy;

import org.json.JSONArray;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.lite.Interpreter;

public class Utils {

    // --------------------------------------------------
    // 1️⃣ Load TFLite model from assets
    // --------------------------------------------------
    public static MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // --------------------------------------------------
    // 2️⃣ Load embeddings.bin + names.json from assets
    // --------------------------------------------------
    public static Pair<float[][], List<String>> loadEmbeddingsAndNames(Context context) {
        float[][] embeddings = null;
        List<String> names = new ArrayList<>();

        try {
            InputStream is = context.getAssets().open("embeddings.bin");
            int size = is.available();
            byte[] bytes = new byte[size];
            int read = is.read(bytes);
            is.close();

            if (read != size) {
                Log.w("Utils", "Warning: Not all bytes were read from embeddings.bin");
            }

            ByteBuffer buffer = ByteBuffer.wrap(bytes);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            int numEmbeddings = buffer.getInt();
            int embeddingSize = buffer.getInt();

            if (numEmbeddings <= 0 || numEmbeddings > 10000 || embeddingSize <= 0 || embeddingSize > 1024) {
                Log.e("Utils", "Invalid embedding dimensions: " + numEmbeddings + " x " + embeddingSize);
                return new Pair<>(new float[0][0], names);
            }

            embeddings = new float[numEmbeddings][embeddingSize];
            for (int i = 0; i < numEmbeddings; i++) {
                for (int j = 0; j < embeddingSize; j++) {
                    embeddings[i][j] = buffer.getFloat();
                }
            }

            InputStream namesStream = context.getAssets().open("names.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(namesStream));
            StringBuilder jsonBuilder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonBuilder.append(line);
            }
            reader.close();
            namesStream.close();

            JSONArray jsonArray = new JSONArray(jsonBuilder.toString());
            for (int i = 0; i < jsonArray.length(); i++) {
                names.add(jsonArray.getString(i));
            }

            Log.i("Utils", "Loaded " + numEmbeddings + " embeddings of size " + embeddingSize);

        } catch (IOException e) {
            Log.e("Utils", "I/O Error loading embeddings: " + e.getMessage());
        } catch (Exception e) {
            Log.e("Utils", "Error parsing embeddings: " + e.getMessage());
        }

        return new Pair<>(embeddings, names);
    }

    // --------------------------------------------------
    // 3️⃣ Convert ImageProxy to Bitmap
    // --------------------------------------------------
    public static Bitmap imageProxyToBitmap(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }

    // --------------------------------------------------
    // 4️⃣ Dummy face detection (placeholder)
    // Replace this with MediaPipe or MLKit logic later
    // --------------------------------------------------
    public static List<Rect> detectFacesMediaPipe(Bitmap bitmap) {
        List<Rect> faces = new ArrayList<>();
        // TODO: integrate real MediaPipe face detector
        // for now, return empty to prevent crash
        return faces;
    }

    // --------------------------------------------------
    // 5️⃣ Extract embedding from face Bitmap
    // --------------------------------------------------
    public static float[] getFaceEmbedding(Bitmap face, Interpreter tflite) {
        int inputSize = 160; // typical FaceNet input size
        Bitmap resized = Bitmap.createScaledBitmap(face, inputSize, inputSize, true);
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());

        int[] intValues = new int[inputSize * inputSize];
        resized.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize);

        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat(((val >> 16) & 0xFF) / 255.0f);
                imgData.putFloat(((val >> 8) & 0xFF) / 255.0f);
                imgData.putFloat((val & 0xFF) / 255.0f);
            }
        }

        float[][] embedding = new float[1][128];
        tflite.run(imgData, embedding);
        return embedding[0];
    }

    // --------------------------------------------------
    // 6️⃣ Compare face embeddings with threshold
    // --------------------------------------------------
    public static String recognizeFace(float[] emb, float[][] knownEmbeddings, List<String> knownNames, float threshold) {
        if (knownEmbeddings == null || knownNames == null || knownEmbeddings.length == 0)
            return "Unknown";

        float minDist = Float.MAX_VALUE;
        int bestIdx = -1;

        for (int i = 0; i < knownEmbeddings.length; i++) {
            float dist = 0;
            for (int j = 0; j < emb.length; j++) {
                float diff = emb[j] - knownEmbeddings[i][j];
                dist += diff * diff;
            }
            dist = (float) Math.sqrt(dist);
            if (dist < minDist) {
                minDist = dist;
                bestIdx = i;
            }
        }

        if (minDist < threshold && bestIdx >= 0) {
            return knownNames.get(bestIdx);
        } else {
            return "Unknown";
        }
    }

    // --------------------------------------------------
    // 7️⃣ Append new embedding (if you save new faces)
    // --------------------------------------------------
    public static boolean appendEmbeddings(Context context, List<float[]> newEmbeddings, String name) {
        // For simplicity, this is just a placeholder
        // You can later write to your own .bin or JSON
        Log.i("Utils", "Appended embedding for: " + name);
        return true;
    }
}
