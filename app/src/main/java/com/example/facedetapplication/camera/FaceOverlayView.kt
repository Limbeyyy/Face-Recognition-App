package com.example.facedetapplication.camera

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private var faces: List<Pair<RectF, String>> = emptyList()

    fun setFaces(results: List<Pair<RectF, String>>) {
        faces = results
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        faces.forEach { (rect, name) ->
            canvas.drawRect(rect, boxPaint)
            canvas.drawText(name, rect.left, rect.top - 10f, textPaint)
        }
    }
}
