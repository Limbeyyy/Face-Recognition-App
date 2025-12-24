package com.example.facedetapplication.util

import androidx.camera.core.ImageProxy
import android.graphics.Bitmap
import android.graphics.YuvImage
import android.graphics.ImageFormat
import java.io.ByteArrayOutputStream
import android.graphics.Rect
import android.graphics.BitmapFactory

object ImageUtils {
    fun imageToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)

        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }
}
