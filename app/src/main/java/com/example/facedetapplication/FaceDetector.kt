package com.example.facedetapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.example.facedetapplication.model.FaceBox
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import kotlin.math.max
import kotlin.math.min

class FaceDetector(context: Context) {

    // Load BlazeFace TFLite model from assets
    private val interpreter: Interpreter = Interpreter(FileUtil.loadMappedFile(context, "blaze_face_short_range.tflite"))

    /**
     * Detect faces in a bitmap and return a list of FaceBox
     */
    fun detect(bitmap: Bitmap): List<FaceBox> {
        val inputSize = 128  // BlazeFace standard input size
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Prepare input tensor [1, H, W, 3] normalized [0,1]
        val inputBuffer = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(3) } } }

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val px = resized.getPixel(x, y)
                inputBuffer[0][y][x][0] = ((px shr 16) and 0xFF) / 255f
                inputBuffer[0][y][x][1] = ((px shr 8) and 0xFF) / 255f
                inputBuffer[0][y][x][2] = (px and 0xFF) / 255f
            }
        }

        // Output placeholders
        val locations = Array(1) { Array(896) { FloatArray(4) } }  // bbox [ymin,xmin,ymax,xmax]
        val confidences = Array(1) { Array(896) { FloatArray(2) } } // score for each box

        val outputs = mapOf(
            0 to locations,
            1 to confidences
        )

        // Run TFLite model
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        val faces = mutableListOf<FaceBox>()
        val widthScale = bitmap.width.toFloat() / inputSize
        val heightScale = bitmap.height.toFloat() / inputSize

        for (i in 0 until 896) {
            val score = confidences[0][i][1]  // object confidence
            if (score < 0.5f) continue  // threshold

            val ymin = max((locations[0][i][0] * inputSize * heightScale).toInt(), 0)
            val xmin = max((locations[0][i][1] * inputSize * widthScale).toInt(), 0)
            val ymax = min((locations[0][i][2] * inputSize * heightScale).toInt(), bitmap.height)
            val xmax = min((locations[0][i][3] * inputSize * widthScale).toInt(), bitmap.width)

            faces.add(FaceBox(Rect(xmin, ymin, xmax, ymax), score))
        }

        return faces
    }

    fun close() {
        interpreter.close()
    }
}
