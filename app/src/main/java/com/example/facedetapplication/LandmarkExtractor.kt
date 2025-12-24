package com.example.facedetapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.pow

class LandmarkExtractor(context: Context) {

    private val interpreter: Interpreter

    init {
        interpreter = Interpreter(loadModelFile(context, "face_landmark.tflite"))
    }

    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        val inputStream = fd.createInputStream()
        val channel = inputStream.channel
        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    /**
     * Extracts raw landmark coordinates from face ROI
     * Output: [x1, y1, x2, y2, ...]
     */
    fun extract(bitmap: Bitmap, bbox: Rect): FloatArray {

        val face = Bitmap.createBitmap(
            bitmap,
            bbox.left.coerceAtLeast(0),
            bbox.top.coerceAtLeast(0),
            bbox.width().coerceAtMost(bitmap.width - bbox.left),
            bbox.height().coerceAtMost(bitmap.height - bbox.top)
        )

        val inputSize = 192 // adjust to your model
        val resized = Bitmap.createScaledBitmap(face, inputSize, inputSize, true)

        val input = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(3) } } }

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val px = resized.getPixel(x, y)
                input[0][y][x][0] = ((px shr 16 and 0xFF) / 255f)
                input[0][y][x][1] = ((px shr 8 and 0xFF) / 255f)
                input[0][y][x][2] = ((px and 0xFF) / 255f)
            }
        }

        // Example: 468 landmarks → 936 floats
        val output = Array(1) { FloatArray(936) }
        interpreter.run(input, output)

        return output[0]
    }

    /**
     * Computes landmark quality score ∈ [0.3, 1.0]
     */
    fun quality(landmarks: FloatArray): Float {

        if (landmarks.isEmpty()) return 0.3f

        // Compute mean X
        var meanX = 0f
        var count = 0
        for (i in landmarks.indices step 2) {
            meanX += landmarks[i]
            count++
        }
        meanX /= count

        // Variance of X
        var variance = 0f
        for (i in landmarks.indices step 2) {
            variance += (landmarks[i] - meanX).pow(2)
        }
        variance /= count

        // Inverse variance → higher = better
        val score = 1f / (1f + variance)

        // Clamp to stable range
        return score.coerceIn(0.3f, 1.0f)
    }
}
