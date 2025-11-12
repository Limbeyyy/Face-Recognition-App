package com.example.facedetproject.UI;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.media.Image;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    // -------- imageProxy to Bitmap (NV21 conversion) ----------
    public static Bitmap imageProxyToBitmap(ImageProxy image) {
        Image img = image.getImage();
        if (img == null) return null;
        // YUV -> NV21 -> JPEG -> Bitmap
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            YuvImageConverter yuvConverter = new YuvImageConverter();
            byte[] nv21 = yuvConverter.yuv420888ToNv21(img);
            int width = image.getWidth();
            int height = image.getHeight();
            android.graphics.YuvImage yuvImage = new android.graphics.YuvImage(nv21,
                    android.graphics.ImageFormat.NV21, width, height, null);
            yuvImage.compressToJpeg(new Rect(0,0,width,height), 90, out);
            byte[] jpeg = out.toByteArray();
            return BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // --------- load TFLite model from assets -------------
    public static MappedByteBuffer loadModelFile(Context ctx, String assetName) throws IOException {
        try (InputStream is = ctx.getAssets().open(assetName)) {
            byte[] bytes = readAllBytes(is);
            ByteBuffer bb = ByteBuffer.allocateDirect(bytes.length);
            bb.order(ByteOrder.nativeOrder());
            bb.put(bytes);
            bb.rewind();
            return bb.asReadOnlyBuffer();
        }
    }

    private static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int r;
        while ((r = is.read(buf)) != -1) bout.write(buf,0,r);
        return bout.toByteArray();
    }

    // --------- Preprocess & run FaceNet to get embedding ----------
    public static float[] getFaceEmbedding(Bitmap faceBmp, Interpreter tflite) {
        Bitmap resized = Bitmap.createScaledBitmap(faceBmp, 160, 160, true);
        // normalize to [-1,1]
        float[][][][] input = new float[1][160][160][3];
        for (int y=0;y<160;y++){
            for (int x=0;x<160;x++){
                int px = resized.getPixel(x,y);
                input[0][y][x][0] = (((px >> 16) & 0xFF) - 127.5f) / 128f;
                input[0][y][x][1] = (((px >> 8) & 0xFF) - 127.5f) / 128f;
                input[0][y][x][2] = ((px & 0xFF) - 127.5f) / 128f;
            }
        }
        float[][] out = new float[1][128];
        tflite.run(input, out);
        return out[0];
    }

    // --------- Compare embedding arrays (Euclidean) ------------
    public static String recognizeFace(float[] embedding, float[][] knownEmbeddings, String[] knownNames, double threshold) {
        if (knownEmbeddings == null || knownEmbeddings.length==0) return "Unknown";
        double minDist = Double.MAX_VALUE;
        int idx = -1;
        for (int i=0;i<knownEmbeddings.length;i++){
            double sum = 0;
            for (int j=0;j<embedding.length;j++){
                double d = embedding[j]-knownEmbeddings[i][j];
                sum += d*d;
            }
            double dist = Math.sqrt(sum);
            if (dist < minDist) { minDist = dist; idx = i; }
        }
        if (minDist <= threshold) return knownNames[idx];
        return "Unknown";
    }

    // ------------ Embeddings binary format load/save helpers ----------------
    // Format (embeddings.bin):
    // int32 N (number of embeddings)
    // int32 D (dimension, e.g., 128)
    // then N * D floats (32-bit little endian)
    // names.json is JSON array of strings

    public static class EmbeddingsData {
        public float[][] embeddings;
        public String[] names;
        public EmbeddingsData(float[][] embeddings, String[] names){ this.embeddings=embeddings; this.names=names; }
    }

    public static EmbeddingsData loadEmbeddingsAndNames(Context ctx, String embAsset, String namesAsset) {
        try {
            // load embeddings
            InputStream is = ctx.getAssets().open(embAsset);
            DataInputStream dis = new DataInputStream(new BufferedInputStream(is));
            int N = dis.readInt();
            int D = dis.readInt();
            float[][] embeddings = new float[N][D];
            for (int i=0;i<N;i++){
                for (int j=0;j<D;j++){
                    embeddings[i][j] = dis.readFloat();
                }
            }
            dis.close();

            // load names json
            InputStream nis = ctx.getAssets().open(namesAsset);
            byte[] nb = readAllBytes(nis);
            String json = new String(nb);
            Gson gson = new Gson();
            TypeToken<String[]> tt = new TypeToken<String[]>(){};
            String[] names = gson.fromJson(json, tt.getType());
            return new EmbeddingsData(embeddings, names);

        } catch (IOException e) {
            e.printStackTrace();
            return new EmbeddingsData(new float[0][0], new String[0]);
        }
    }

    // Append embeddings + name to existing stored binary (used in Register)
    // This reads existing embeddings.bin + names.json and writes new combined files into internal storage then copies to assets if needed.
    public static boolean appendEmbeddings(Context ctx, List<float[]> newEmb, String newName) {
        try {
            EmbeddingsData data = loadEmbeddingsAndNames(ctx, "embeddings.bin", "names.json");
            int oldN = data.embeddings.length;
            int D = oldN==0 ? newEmb.get(0).length : data.embeddings[0].length;
            int newN = oldN + newEmb.size();

            float[][] merged = new float[newN][D];
            // copy old
            for (int i=0;i<oldN;i++) merged[i] = data.embeddings[i];
            for (int i=0;i<newEmb.size();i++) merged[oldN + i] = newEmb.get(i);

            // names: load old names.json
            // Read old names
            InputStream nis = ctx.getAssets().open("names.json");
            byte[] nb = readAllBytes(nis);
            String json = new String(nb);
            Gson g = new Gson();
            TypeToken<String[]> tt = new TypeToken<String[]>(){};
            String[] oldNames = g.fromJson(json, tt.getType());
            String[] mergedNames = new String[oldNames.length + 1];
            System.arraycopy(oldNames, 0, mergedNames, 0, oldNames.length);
            mergedNames[mergedNames.length-1] = newName;

            // Save into internal files (app-private)
            // embeddings.bin
            java.io.File outEmb = new java.io.File(ctx.getFilesDir(), "embeddings.bin");
            try (DataOutputStream dos = new DataOutputStream(new java.io.BufferedOutputStream(new java.io.FileOutputStream(outEmb)))) {
                dos.writeInt(merged.length);
                dos.writeInt(D);
                for (int i=0;i<merged.length;i++){
                    for (int j=0;j<D;j++) dos.writeFloat(merged[i][j]);
                }
            }

            // names.json
            java.io.File outNames = new java.io.File(ctx.getFilesDir(), "names.json");
            try (java.io.FileWriter fw = new java.io.FileWriter(outNames)) {
                g.toJson(mergedNames, fw);
            }

            // Note: App will need to read from internal storage if present; modify loader to check internal files first
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    // ------------ MediaPipe detection placeholder -------------
    // Implement this by integrating MediaPipe's Android face detector.
    // Return list of Rect objects in pixel coordinates w.r.t the bitmap.
    // For now, this is a stub that returns empty list.
    public static List<Rect> detectFacesMediaPipe(Bitmap bitmap) {
        // TODO: Replace this stub with actual MediaPipe detection code.
        // Example plan:
        // 1. Create FaceDetectorOptions and FaceDetector as per MediaPipe Tasks API.
        // 2. Run detector.detect(bitmap) or detector.detectForVideo(...) according to API.
        // 3. For each detected face, compute bounding box in pixel coordinates and add to list.
        //
        // See MediaPipe Android Tasks documentation for exact code. The dependency is:
        // implementation 'com.google.mediapipe:mediapipe-face-detection:0.9.0'
        return new ArrayList<>();
    }

    // small utility class to convert Image to NV21
    private static class YuvImageConverter {
        // Convert Image (YUV_420_888) to NV21 byte[]
        public byte[] yuv420888ToNv21(Image image) {
            Image.Plane[] planes = image.getPlanes();
            int width = image.getWidth();
            int height = image.getHeight();
            byte[] nv21;
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            android.graphics.YuvImage yuvImage;
            try {
                java.nio.ByteBuffer yBuffer = planes[0].getBuffer();
                java.nio.ByteBuffer uBuffer = planes[1].getBuffer();
                java.nio.ByteBuffer vBuffer = planes[2].getBuffer();

                int ySize = yBuffer.remaining();
                int uSize = uBuffer.remaining();
                int vSize = vBuffer.remaining();

                nv21 = new byte[ySize + uSize + vSize];
                yBuffer.get(nv21, 0, ySize);
                vBuffer.get(nv21, ySize, vSize);
                uBuffer.get(nv21, ySize + vSize, uSize);
                return nv21;
            } catch (Exception e) {
                e.printStackTrace();
                return new byte[0];
            }
        }
    }
}
