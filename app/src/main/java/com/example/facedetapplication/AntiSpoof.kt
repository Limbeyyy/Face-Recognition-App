package com.example.facedetapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import androidx.core.graphics.scale

class AntiSpoof(
    context: Context,
    modelFile: String,
    private val inputSize: Int = 224    // change if model differs
) {

    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context, modelFile)
        interpreter = Interpreter(model)
    }

    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        fd.createInputStream().use { input ->
            val channel = input.channel
            return channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        }
    }

    /**
     * @return confidence ∈ [0,1]
     * 1.0 → real face
     * 0.0 → spoof
     */
    fun predict(frame: Bitmap, bbox: Rect): Float {
        val cropped = cropSafe(frame, bbox)
        val resized = cropped.scale(inputSize, inputSize)

        val input = preprocess(resized)
        val output = Array(1) { FloatArray(1) }

        interpreter.run(input, output)

        return output[0][0].coerceIn(0f, 1f)
    }

    // -------------------------------
    // Helpers
    // -------------------------------

    private fun cropSafe(bitmap: Bitmap, rect: Rect): Bitmap {
        val left = max(0, rect.left)
        val top = max(0, rect.top)
        val right = min(bitmap.width, rect.right)
        val bottom = min(bitmap.height, rect.bottom)

        val width = max(1, right - left)
        val height = max(1, bottom - top)

        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    /**
     * Converts Bitmap → Float32 tensor [1, H, W, 3]
     * Normalized to [0,1]
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(
            pixels,
            0,
            inputSize,
            0,
            0,
            inputSize,
            inputSize
        )

        pixels.forEach { pixel ->
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)  // G
            buffer.putFloat((pixel and 0xFF) / 255f)         // B
        }

        buffer.rewind()
        return buffer
    }

    fun close() {
        interpreter.close()
    }
}
