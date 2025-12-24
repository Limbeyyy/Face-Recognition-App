package com.example.facedetapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min

class FaceRecognition(context: Context, modelFile: String = "facenet.tflite") {

    private val interpreter: Interpreter
    private val inputSize = 160  // typical FaceNet input size

    init {
        interpreter = Interpreter(loadModelFile(context, modelFile))
    }

    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        val inputStream = fd.createInputStream()
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    /**
     * Embed a face inside bbox
     * @param bitmap - full frame
     * @param bbox - detected face bounding box
     * @return embedding FloatArray (512-dim for FaceNet)
     */
    fun embed(bitmap: Bitmap, bbox: Rect): FloatArray {

        // 1. Crop face safely
        val left = bbox.left.coerceAtLeast(0)
        val top = bbox.top.coerceAtLeast(0)
        val right = min(bitmap.width, bbox.right)
        val bottom = min(bitmap.height, bbox.bottom)
        val face = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)

        // 2. Resize to model input
        val resized = Bitmap.createScaledBitmap(face, inputSize, inputSize, true)

        // 3. Preprocess: Float32 tensor [1, H, W, 3], normalized [-1,1] for FaceNet
        val input = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        input.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (px in pixels) {
            val r = ((px shr 16) and 0xFF) / 127.5f - 1f
            val g = ((px shr 8) and 0xFF) / 127.5f - 1f
            val b = ((px) and 0xFF) / 127.5f - 1f
            input.putFloat(r)
            input.putFloat(g)
            input.putFloat(b)
        }
        input.rewind()

        // 4. Run model
        val embedding = Array(1) { FloatArray(512) }  // FaceNet 512-dim
        interpreter.run(input, embedding)

        return embedding[0]
    }

    fun close() {
        interpreter.close()
    }
}
