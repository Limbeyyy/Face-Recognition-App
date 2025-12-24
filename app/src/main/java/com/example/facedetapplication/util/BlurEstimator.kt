package com.example.facedetapplication.util

import android.graphics.Bitmap
import android.graphics.Rect
import kotlin.math.min

object BlurEstimator {

    /**
     * Computes a blur score for a face region in a bitmap
     * Higher score = sharper image, lower = blurry
     * Output normalized between 0f..1f
     */
    fun score(bitmap: Bitmap, bbox: Rect): Float {
        // Safe crop
        val left = bbox.left.coerceAtLeast(0)
        val top = bbox.top.coerceAtLeast(0)
        val right = min(bitmap.width, bbox.right)
        val bottom = min(bitmap.height, bbox.bottom)

        val width = right - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return 0.3f

        val faceRegion = Bitmap.createBitmap(bitmap, left, top, width, height)

        // Convert pixels to grayscale intensity
        val pixels = IntArray(width * height)
        faceRegion.getPixels(pixels, 0, width, 0, 0, width, height)

        var sum = 0f
        var sumSq = 0f
        for (p in pixels) {
            val gray = ((p shr 16 and 0xFF) +
                    (p shr 8 and 0xFF) +
                    (p and 0xFF)) / 3f
            sum += gray
            sumSq += gray * gray
        }

        val mean = sum / pixels.size
        val variance = sumSq / pixels.size - mean * mean

        // Normalize variance (tweak 1000f to calibrate)
        return (variance / 1000f).coerceIn(0.3f, 1f)
    }
}
